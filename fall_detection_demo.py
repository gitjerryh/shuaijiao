import argparse
import sys
import time
from collections import deque
from functools import lru_cache

import cv2
import numpy as np

from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import (NetworkIntrinsics,
                                      postprocess_nanodet_detection)
import serial

# 用于存储当前的检测结果
last_detections = []
# 用于存储历史边界框数据，每个人体一个队列
person_history = {}
# 用于存储人体失踪帧计数
frames_missing = {}
# 人体ID计数器
next_person_id = 0
# 摔倒状态
fall_detected = False
# 摔倒警报时间戳
fall_alert_time = 0
# 配置参数
HISTORY_SIZE = 30  # 跟踪历史帧数
FALL_THRESHOLD_RATIO = 0.8  # 宽高比阈值
FALL_THRESHOLD_VELOCITY = 0.05  # 垂直速度阈值
PERSON_LOST_FRAMES = 10  # 人体消失多少帧后移除追踪
MIN_TRACKING_FRAMES = 10  # 最小跟踪帧数才开始判断
IOU_THRESHOLD = 0.3  # 边界框匹配IOU阈值


class Detection:
    def __init__(self, coords, category, conf, metadata):
        """创建检测对象，记录边界框、类别和置信度"""
        self.category = category
        self.conf = conf
        self.box = imx500.convert_inference_coords(coords, metadata, picam2)
        self.person_id = -1  # 初始未分配ID


def parse_detections(metadata: dict):
    """解析输出张量为多个检测对象，缩放到ISP输出"""
    global last_detections
    bbox_normalization = intrinsics.bbox_normalization
    bbox_order = intrinsics.bbox_order
    threshold = args.threshold
    iou = args.iou
    max_detections = args.max_detections

    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    input_w, input_h = imx500.get_input_size()
    if np_outputs is None:
        return last_detections
    if intrinsics.postprocess == "nanodet":
        boxes, scores, classes = \
            postprocess_nanodet_detection(outputs=np_outputs[0], conf=threshold, iou_thres=iou,
                                          max_out_dets=max_detections)[0]
        from picamera2.devices.imx500.postprocess import scale_boxes
        boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
    else:
        boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
        if bbox_normalization:
            boxes = boxes / input_h

        if bbox_order == "xy":
            boxes = boxes[:, [1, 0, 3, 2]]
        boxes = np.array_split(boxes, 4, axis=1)
        boxes = zip(*boxes)

    # 创建所有检测对象
    detections = [
        Detection(box, category, score, metadata)
        for box, score, category in zip(boxes, scores, classes)
        if score > threshold
    ]
    
    # 只保留人体检测（类别0通常是person）
    labels = get_labels()
    person_detections = []
    for det in detections:
        label = labels[int(det.category)]
        if label.lower() == "person":
            person_detections.append(det)
    
    last_detections = person_detections
    return last_detections


def calculate_iou(box1, box2):
    """计算两个边界框的IoU"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # 计算相交区域
    xx1 = max(x1, x2)
    yy1 = max(y1, y2)
    xx2 = min(x1 + w1, x2 + w2)
    yy2 = min(y1 + h1, y2 + h2)
    
    # 计算相交区域面积
    w = max(0, xx2 - xx1)
    h = max(0, yy2 - yy1)
    inter_area = w * h
    
    # 计算各自区域面积
    box1_area = w1 * h1
    box2_area = w2 * h2
    
    # 计算IoU
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou


def update_tracking(detections):
    """更新人体跟踪状态"""
    global person_history, frames_missing, next_person_id
    
    # 为每个检测到的人体分配ID
    unmatched_detections = []
    for det in detections:
        matched = False
        # 尝试匹配现有ID
        for person_id in list(person_history.keys()):
            if not person_history[person_id]:  # 空历史记录，跳过
                continue
            last_box = person_history[person_id][-1]
            if calculate_iou(det.box, last_box) > IOU_THRESHOLD:
                det.person_id = person_id
                person_history[person_id].append(det.box)
                # 保持历史记录长度
                if len(person_history[person_id]) > HISTORY_SIZE:
                    person_history[person_id].popleft()
                matched = True
                break
        
        if not matched:
            unmatched_detections.append(det)
    
    # 为未匹配的检测创建新ID
    for det in unmatched_detections:
        det.person_id = next_person_id
        person_history[next_person_id] = deque([det.box], maxlen=HISTORY_SIZE)
        frames_missing[next_person_id] = 0
        next_person_id += 1
    
    # 移除长时间未匹配的人体
    active_ids = set(det.person_id for det in detections)
    for person_id in list(person_history.keys()):
        if person_id not in active_ids:
            # 记录该人体未被检测到的帧数
            if person_id not in frames_missing:
                frames_missing[person_id] = 1
            else:
                frames_missing[person_id] += 1
            
            # 如果连续多帧未检测到，则移除
            if frames_missing[person_id] > PERSON_LOST_FRAMES:
                del person_history[person_id]
                del frames_missing[person_id]
        else:
            # 重置未检测计数器
            frames_missing[person_id] = 0


def detect_fall():
    """检测是否发生摔倒"""
    global fall_detected, fall_alert_time
    
    fall_status = False
    
    # 分析每个人体的历史数据
    for person_id, history in person_history.items():
        # 至少需要一定数量的历史帧
        if len(history) < MIN_TRACKING_FRAMES:
            continue
        
        # 获取前一段时间内的边界框
        recent_boxes = list(history)
        
        # 计算前后的宽高比变化
        first_boxes = recent_boxes[:3]  # 前3帧
        last_boxes = recent_boxes[-3:]  # 后3帧
        
        # 计算平均宽高比
        first_ratios = [box[2] / box[3] for box in first_boxes]
        last_ratios = [box[2] / box[3] for box in last_boxes]
        first_avg_ratio = sum(first_ratios) / len(first_ratios)
        last_avg_ratio = sum(last_ratios) / len(last_ratios)
        
        # 计算中心点Y坐标变化（垂直速度）
        first_y = [(box[1] + box[3]/2) for box in first_boxes]
        last_y = [(box[1] + box[3]/2) for box in last_boxes]
        first_avg_y = sum(first_y) / len(first_y)
        last_avg_y = sum(last_y) / len(last_y)
        
        # 获取图像高度
        height = picam2.camera_properties["ScalerCropMaximum"][3]
        vertical_velocity = (last_avg_y - first_avg_y) / height
        
        # 判断摔倒条件
        # 1. 宽高比增大（人体变为更宽的形状）
        # 2. 垂直位置有明显下降
        ratio_change = last_avg_ratio - first_avg_ratio
        
        if ratio_change > FALL_THRESHOLD_RATIO and vertical_velocity > FALL_THRESHOLD_VELOCITY:
            fall_status = True
            fall_detected = True
            fall_alert_time = time.time()
            ser.write(b'FALL\n')  # 通过串口发送摔倒消息
            break
    
    # 如果摔倒报警超过5秒，重置状态
    if fall_detected and (time.time() - fall_alert_time) > 5:
        fall_detected = False
    
    return fall_status


@lru_cache
def get_labels():
    """获取标签列表"""
    labels = intrinsics.labels

    if intrinsics.ignore_dash_labels:
        labels = [label for label in labels if label and label != "-"]
    return labels


def draw_detections(request, stream="main"):
    """在ISP输出上绘制检测结果"""
    detections = last_results
    if detections is None:
        return
    labels = get_labels()
    
    with MappedArray(request, stream) as m:
        # 绘制边界框
        for detection in detections:
            x, y, w, h = detection.box
            label = f"Person #{detection.person_id} ({detection.conf:.2f})"

            # 计算文本大小和位置
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_x = x + 5
            text_y = y + 15

            # 创建一个数组副本，用于绘制带有透明度的背景
            overlay = m.array.copy()

            # 在覆盖层上绘制背景矩形
            cv2.rectangle(overlay,
                          (text_x, text_y - text_height),
                          (text_x + text_width, text_y + baseline),
                          (255, 255, 255),  # 背景颜色（白色）
                          cv2.FILLED)

            alpha = 0.30
            cv2.addWeighted(overlay, alpha, m.array, 1 - alpha, 0, m.array)

            # 在背景上绘制文本
            cv2.putText(m.array, label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # 绘制检测框
            if detection.person_id in person_history and len(person_history[detection.person_id]) >= MIN_TRACKING_FRAMES:
                # 跟踪时间足够长的人体用绿色框
                cv2.rectangle(m.array, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
            else:
                # 新跟踪的人体用蓝色框
                cv2.rectangle(m.array, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)
        
        # 绘制摔倒警报
        if fall_detected:
            # 显示摔倒警告
            warning_text = "!!! FALL DETECTED !!!"
            (text_width, text_height), _ = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            text_x = (m.array.shape[1] - text_width) // 2
            text_y = 50
            
            # 绘制半透明背景
            overlay = m.array.copy()
            cv2.rectangle(overlay,
                          (text_x - 10, text_y - text_height - 10),
                          (text_x + text_width + 10, text_y + 10),
                          (0, 0, 255),
                          cv2.FILLED)
            
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, m.array, 1 - alpha, 0, m.array)
            
            # 绘制警告文本
            cv2.putText(m.array, warning_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)


def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="模型路径",
                        default="/usr/share/imx500-models/imx500_network_nanodet_plus_416x416_pp.rpk")
    parser.add_argument("--fps", type=int, help="每秒帧数")
    parser.add_argument("--bbox-normalization", action=argparse.BooleanOptionalAction, help="归一化边界框")
    parser.add_argument("--bbox-order", choices=["yx", "xy"], default="yx",
                        help="设置边界框顺序 yx -> (y0, x0, y1, x1) xy -> (x0, y0, x1, y1)")
    parser.add_argument("--threshold", type=float, default=0.55, help="检测阈值")
    parser.add_argument("--iou", type=float, default=0.65, help="设置IoU阈值")
    parser.add_argument("--max-detections", type=int, default=10, help="设置最大检测数")
    parser.add_argument("--ignore-dash-labels", action=argparse.BooleanOptionalAction, help="移除'-'标签")
    parser.add_argument("--postprocess", choices=["", "nanodet"],
                        default=None, help="运行指定类型的后处理")
    parser.add_argument("-r", "--preserve-aspect-ratio", action=argparse.BooleanOptionalAction,
                        help="保持输入张量的像素宽高比")
    parser.add_argument("--labels", type=str,
                        help="标签文件路径")
    parser.add_argument("--print-intrinsics", action="store_true",
                        help="打印JSON网络内部参数然后退出")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # 必须在Picamera2实例化前调用
    imx500 = IMX500(args.model)
    intrinsics = imx500.network_intrinsics
    if not intrinsics:
        intrinsics = NetworkIntrinsics()
        intrinsics.task = "object detection"
    elif intrinsics.task != "object detection":
        print("网络不是物体检测任务", file=sys.stderr)
        exit()

    # 从参数覆盖内部参数
    for key, value in vars(args).items():
        if key == 'labels' and value is not None:
            with open(value, 'r') as f:
                intrinsics.labels = f.read().splitlines()
        elif hasattr(intrinsics, key) and value is not None:
            setattr(intrinsics, key, value)

    # 默认值
    if intrinsics.labels is None:
        with open("assets/coco_labels.txt", "r") as f:
            intrinsics.labels = f.read().splitlines()
    intrinsics.update_with_defaults()

    if args.print_intrinsics:
        print(intrinsics)
        exit()

    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(controls={"FrameRate": intrinsics.inference_rate}, buffer_count=12)

    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=True)

    # 初始化串口连接ESP32的Serial2
    # 树莓派的GPIO14(TX)连接ESP32的RXD2(16)
    # 树莓派的GPIO15(RX)连接ESP32的TXD2(17)
    ser = serial.Serial('/dev/ttyAMA10', 9600, timeout=1)  
    print("串口初始化完成，准备向ESP32发送摔倒检测信息")

    if intrinsics.preserve_aspect_ratio:
        imx500.set_auto_aspect_ratio()

    last_results = None
    picam2.pre_callback = draw_detections
    
    print("摔倒检测系统启动中...")
    print("按Ctrl+C退出程序")
    
    try:
        while True:
            # 解析检测结果
            last_results = parse_detections(picam2.capture_metadata())
            # 更新跟踪
            update_tracking(last_results)
            # 检测摔倒
            if detect_fall():
                print("警告：检测到摔倒！")
            time.sleep(0.01)  # 添加小延迟以避免CPU占用过高
    except KeyboardInterrupt:
        print("程序已退出")

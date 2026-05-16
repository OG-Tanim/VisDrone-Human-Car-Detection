from ultralytics import YOLO
import cv2
import numpy as np
import argparse
import os


def predict(
    source,
    conf=0.25,
    imgsz=640,
    device="cpu",
    stream=False,
    save=False,
    model_path="weights/YOLO-26s_human_car_model.pt",
    human_class_id=0,
    car_class_id=1,
):
    model = YOLO(model_path)
    results = model.predict(
        source=source, conf=conf, imgsz=imgsz, device=device, stream=stream, save=False, verbose=False
    )

    human_counts, car_counts = [], []

    for i, r in enumerate(results):
        human_count = int((r.boxes.cls == human_class_id).sum())
        car_count = int((r.boxes.cls == car_class_id).sum())
        human_counts.append(human_count)
        car_counts.append(car_count)

        annotated = r.plot()
        font = cv2.FONT_HERSHEY_SIMPLEX
        h, w = annotated.shape[:2]
        margin = 15
        line_height = 40

        text_humans = f"Humans: {human_count}"
        text_cars = f"Cars: {car_count}"

        (tw_h, _), _ = cv2.getTextSize(text_humans, font, 1.25, 3)
        (tw_c, _), _ = cv2.getTextSize(text_cars, font, 1.25, 3)
        box_w = max(tw_h, tw_c) + 2 * margin
        box_h = 2 * line_height + 3 * margin

        overlay = annotated.copy()
        cv2.rectangle(overlay, (margin, margin), (margin + box_w, margin + box_h), (0, 0, 0), -1)
        annotated = cv2.addWeighted(overlay, 0.5, annotated, 0.5, 0)

        cv2.putText(annotated, text_humans, (margin + 10, margin + 30), font, 1.25, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.putText(annotated, text_cars, (margin + 10, margin + 30 + line_height), font, 1.25, (255, 0, 0), 3, cv2.LINE_AA)

        out_dir = "runs/predict"
        run = 1
        while os.path.exists(f"{out_dir}/run_{run:03d}"):
            run += 1
        out_dir = f"{out_dir}/run_{run:03d}"
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(source) if os.path.isfile(source) else f"result_{i}")[0]
        out_path = os.path.join(out_dir, f"{stem}_annotated.jpg")
        cv2.imwrite(out_path, annotated)

        if save:
            print(f"Saved to {out_path}")
        else:
            os.remove(out_path)

        cv2.namedWindow("YOLO26 Prediction", cv2.WINDOW_NORMAL)
        cv2.imshow("YOLO26 Prediction", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return human_counts, car_counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO26 Human/Car Detection")
    parser.add_argument("source", type=str, help="Input source (image, video, URL, directory, or webcam index)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--device", type=str, default="cpu", help="Computational device (e.g. 0, cpu)")
    parser.add_argument("--stream", action="store_true", help="Use streaming generator for long videos")
    parser.add_argument("--save", action="store_true", help="Save annotated images to runs/predict/")
    parser.add_argument("--model", type=str, default="weights/YOLO-26s_human_car_model.pt", help="Path to model weights")
    args = parser.parse_args()

    predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        stream=args.stream,
        save=args.save,
        model_path=args.model,
    )

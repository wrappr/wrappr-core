import argparse
from sys import platform, argv

from models import *  # set ONNX_EXPORT in models.py
import utils.datasets
import utils.utils


def detect(id, filename, data='data/coco.data', cfg='cfg/yolov3-spp.cfg', weights='weights/yolov3-spp.weights', half=False, img_size=416, conf_thres=0.3, nms_thres=0.5):
    with torch.no_grad():
        # Initialize
        device = torch_utils.select_device('') # XXX

        # Initialize model
        model = Darknet(cfg, img_size)

        # Load weights
        if weights.endswith('.pt'):  # pytorch format
            model.load_state_dict(torch.load(weights, map_location=device)['model'])
        else:  # darknet format
            _ = load_darknet_weights(model, weights)

        # Fuse Conv2d + BatchNorm2d layers
        # model.fuse()

        # Eval mode
        model.to(device).eval()

        # Half precision
        half = half and device.type != 'cpu'  # half precision only supported on CUDA
        if half:
            model.half()

        # Set Dataloader
        save_img = True
        dataset = utils.datasets.LoadImages(filename, img_size=img_size, half=half)

        # Get classes and colors
        classes = load_classes(parse_data_cfg(data)['names'])
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

        # Run inference
        t0 = time.time()

        path, img, im0s, _ = next(dataset.__iter__())
        # Get detections
        img = torch.from_numpy(img).to(device)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred, _ = model(img)

        if half:
            pred = pred.float()

        objects = []
        for i, det in enumerate(utils.utils.non_max_suppression(pred, conf_thres, nms_thres)):  # detections per image
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class

                # Draw boxes
                for *xyxy, conf, _, cls in det:
                    label = classes[int(cls)]
                    plot_one_box(xyxy, im0s, label=label, color=colors[int(cls)])
                    objects.append((list(map(int, xyxy)), label, float(conf)))


    return id, im0s, 0, objects

if __name__ == '__main__':
        id, img, score, objects = detect(id, argv[1])
        print(objects)
        cv2.imshow('test', img)
        cv2.waitKey()

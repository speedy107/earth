# Earth — phone camera block detector

Open **https://speedy107.github.io/earth/** on your phone in Chrome or Safari. Wait for Ready, tap Start camera and allow access. The rear camera is preferred. Stop releases it. Choose photo runs the same detector on a local image without camera permission.

Inference runs locally in your browser using ONNX Runtime Web 1.20.1 (single-thread WASM). Frames and photos are not uploaded. The first load downloads the 10 MB FP32 model plus the runtime from jsDelivr. Internet access is needed to load these resources. Speed depends on the device; no phone FPS guarantee is made.

Model: YOLO11n; input RGB float32 [1,3,320,320], letterbox padding 114 and normalization /255. Output [1,6,2100] has pixel xywh plus blue/red scores, no objectness. The app applies confidence filtering and class-aware NMS at 0.45, then reverses letterboxing. Class 0 = blue_box, class 1 = red_box.

The experimental model's supplied metadata reports 53 images, 45 train and 8 validation. Reported mAP50 0.995 is a small model-selection result, not field accuracy. Test unfamiliar scenes, distant objects, both colors together and false positives before robot use.

Download earth-training-kit.zip for the original model package, dataset validator, augmentation plan and training configurations. Original source images and labels were not present in the uploaded model archive.

GitHub Pages publishes the main branch root. To run locally, serve this folder with an HTTP server and open localhost on the same computer. Phone camera access over a LAN requires HTTPS; the GitHub Pages URL supplies HTTPS.

References: https://onnxruntime.ai/docs/get-started/with-javascript/web.html and https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site

# CHANGELOG for Fast-Track

## Known Issues:
- See GitHub Issues

## v1.3.0 - Nate Haddad, 4/1/2025
- Rewrite database code

## v1.2.2 - Nate Haddad, 3/31/2025
- Update track messages

## v1.2.1 - Nate Haddad, 3/31/2025
- Update image captions

## v1.2.0 - Nate Haddad, 3/27/2025
- Add `RFDETR` to `detectors`
- Update Python packages and versions
- Add ruff formatting
- Minor refactor

## v1.1.1 - Nate Haddad, 2/25/2024
- Add support for class selection (fixes bug)

## v1.1.0 - Nate Haddad, 2/25/2024
- Add `YOLOv9ONNX` to `detectors`

## v1.0.0 - Nate Haddad, 2/19/2024
- Add database option to tracker
- Add `SQLDatabase` database class
- Major refactor of all modules
- Additional options for `OpenAI` image processing

## v0.8.0 - Nate Haddad, 2/13/2024
- Update `BYTETracker` with a variety of new features:
  - add `crops` list, which adds a crop of the object to the track state when it is created and removed
  - add track messages, for relaying information to a database or information store

## v0.7.0 - Nate Haddad, 2/12/2024
- Refactor `object_detection` and `object_tracking` modules into `detectors` and `trackers` modules
- Refactor `YOLOV8ONNX` into `yolo_v8_onnx.py`

## v0.6.2 - Nate Haddad, 2/12/2024
- Update dependencies

## v0.6.1 - Nate Haddad, 8/19/2023
- Add YOLOv8 L to demo and pretrained models

## v0.6.0 - Nate Haddad, 8/19/2023
- Update Gradio app
- Add support for loading custom models with demo app

## v0.5.0 - Nate Haddad, 8/19/2023
- Add Ultralytics YOLOv8
- Add Gradio demo with support for YOLO-NAS and YOLOv8

## v0.4.0 - Nate Haddad, 5/13/2023
- Make tracking optional
- Refactor video writing
- Add GitHub Actions pipeline

## v0.3.3 - Nate Haddad, 5/13/2023
- Fix `py-modules` bug in `pyproject.toml`

## v0.3.2 - Nate Haddad, 5/9/2023
- Enables CPU and GPU use of YOLO-NAS

## v0.3.1 - Nate Haddad, 5/6/2023
- Clean up `YOLONAS` `detect` method

## v0.3.0 - Nate Haddad, 5/5/2023
- Add support for `YOLO-NAS` models
- Fix text visualization alignment

## v0.2.0 - Nate Haddad, 1/24/2023
- Add support for `YOLOv8` ONNX models
- Refactor `object_detection` module
- Refactor `object_tracking` module
- Add `.pylintrc`
- Add project version

## v0.1.1 - Nate Haddad, 1/21/2023
- Fix track visualization (visualizes tracks, not only track updates)

## v0.1.0 - Nate Haddad, 12/30/2022
- Project initialization
- implement object tracking pipeline

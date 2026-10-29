from roboflow import Roboflow

rf = Roboflow(api_key="tRVHHVVhltdOXqileCgJ")
project = rf.workspace("jvck-thwala").project("mineguard-detection-crele")
version = project.version(7)  # Your Roboflow dataset version number

# Uploads your local weights/best.pt file to Roboflow's servers
version.deploy(
    model_type="yolov8",
    model_path="C:/Users/231348700/Desktop/work/MineGuard.DetectionAI/",
    filename="best.pt"
)
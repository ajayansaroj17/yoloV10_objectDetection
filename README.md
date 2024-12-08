
# Object Detection Microservice  

This repository implements an object detection microservice using the YOLOv10 model with pretrained weights on the COCO dataset. The microservice processes uploaded images, detects objects, and returns predictions in a structured JSON format.

---

## **Features**  

- Upload an image for object detection.  
- Automatic resizing, transformation, and preprocessing of images.  
- Inference using the YOLOv10 model.  
- Prediction results saved as `predict.json`.  
- Visualization of detection results via a web interface on port `5000`.  

---

## **Model Reference**  

- **YOLOv10 Object Detection**: [GitHub Repository](https://github.com/THU-MIG/yolov10)  

---

## **Getting Started**  

### **1. Clone the Repository**  
```bash  
git clone <repo-url>  
cd <repo-directory>  
```

### **2. Install Dependencies**  
```bash  
pip install -r requirements.txt  
```

### **3. Run the Application**  
```bash  
python app.py  
```

---

## **How It Works**  

1. **Upload Image**: Upload an image through the web interface.  
2. **Preprocessing**: The uploaded image is resized, transformed, and preprocessed.  
3. **Model Inference**: The YOLOv10 model predicts objects in the image.  
4. **Prediction Output**: The predictions are saved as `predict.json` and displayed on the web interface.  

---

## **Usage Example**  

- **Access the Application**:  
  Open [http://localhost:5000](http://localhost:5000) in your web browser.  

- **Upload Image**:  
  Use the provided upload feature to select an image.  

- **View Results**:  
  See detected objects on the interface and inspect the generated `predict.json` file for detailed results.  

---

## **Contributing**  

We welcome contributions! Please submit issues or create pull requests to help improve the project.

---

## **License**  

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## **Acknowledgements**  

- [YOLOv10 Object Detection](https://github.com/THU-MIG/yolov10)  
- COCO Dataset for pre-trained model weights.  

import streamlit as st
import cv2
import numpy as np
import base64
import logging
import json
import requests
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from PIL import Image
import io
import os
from ultralytics import YOLO
from google import genai
from google.genai import types as genai_types
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('threat_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ConfigManager:
    def __init__(self):
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.model_path = 'yolov8n.pt'  
        
    def get_api_key_from_user(self) -> str:
        """Get API key from user input in sidebar if not available in environment"""
        if 'gemini_api_key' not in st.session_state:
            st.session_state.gemini_api_key = self.gemini_api_key or ""
        
        with st.sidebar:
            st.header("🔑 API Configuration")
            
            if not self.gemini_api_key:
                st.warning("⚠️ Gemini API key not found in environment")
                
                api_key_input = st.text_input(
                    "Enter your Gemini API Key:",
                    type="password",
                    value=st.session_state.gemini_api_key,
                    placeholder="Your Google Gemini API Key",
                    help="Get your API key from https://makersuite.google.com/app/apikey",
                    key="gemini_api_key_input"
                )
                
                if api_key_input and api_key_input != st.session_state.gemini_api_key:
                    st.session_state.gemini_api_key = api_key_input
                    # Invalidate cached validation so it re-runs with the new key
                    st.session_state.pop('config_validated', None)
                    st.session_state.pop('config_api_key', None)
                    st.success("✅ API Key updated!")
                    st.rerun()  # Refresh to validate the new key
                
                return st.session_state.gemini_api_key
            else:
                st.success("✅ API Key loaded from environment")
                return self.gemini_api_key
        
    def validate_config(self) -> tuple[bool, str]:
        """Validate configuration and return (is_valid, api_key)"""
        # Return cached result if already validated this run
        if 'config_validated' in st.session_state and 'config_api_key' in st.session_state:
            return st.session_state.config_validated, st.session_state.config_api_key

        api_key = self.get_api_key_from_user()
        
        if not api_key:
            with st.sidebar:
                st.error("❌ Please provide a valid Gemini API Key")
            st.session_state.config_validated = False
            st.session_state.config_api_key = ""
            return False, ""
        
        # Test the API key validity
        try:
            client = genai.Client(api_key=api_key)
            # Try a lightweight call to test the key
            client.models.generate_content(
                model='gemini-3-flash-preview',
                contents='Test'
            )
            with st.sidebar:
                st.success("✅ API Key validated successfully")
            st.session_state.config_validated = True
            st.session_state.config_api_key = api_key
            return True, api_key
        except Exception as e:
            with st.sidebar:
                st.error(f"❌ Invalid API Key: {str(e)}")
            st.session_state.config_validated = False
            st.session_state.config_api_key = ""
            return False, ""

class ObjectDetector:
    def __init__(self, model_path: str):
        try:
            logger.info(f"Loading YOLO model from {model_path}...")
            self.model = YOLO(model_path)
            import numpy as np
            dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = self.model(dummy_image, verbose=False)
            logger.info(f"YOLO model loaded and tested successfully from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {str(e)}")
            alternative_models = ['yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt']
            for alt_model in alternative_models:
                try:
                    logger.info(f"Trying alternative model: {alt_model}")
                    self.model = YOLO(alt_model)
                    _ = self.model(dummy_image, verbose=False)
                    logger.info(f"Successfully loaded alternative model: {alt_model}")
                    break
                except Exception as alt_e:
                    logger.warning(f"Alternative model {alt_model} also failed: {str(alt_e)}")
                    continue
            else:
                raise Exception(f"All YOLO models failed to load. Original error: {str(e)}")
        if not hasattr(self, 'model') or self.model is None:
            raise Exception("YOLO model initialization failed completely")
    
    def detect_objects(self, image: np.ndarray, confidence_threshold: float = 0.5) -> List[Dict]:
        try:
            if image is None or image.size == 0:
                logger.error("Input image is None or empty")
                return []
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            elif len(image.shape) == 3 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            logger.info(f"Running YOLO inference on image of shape: {image.shape}")
            results = self.model(image, conf=confidence_threshold, verbose=False)
            detected_objects = []
            if results is not None and len(results) > 0:
                for result in results:
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        boxes = result.boxes
                        for i in range(len(boxes)):
                            try:
                                confidence = float(boxes.conf[i]) if boxes.conf is not None else 0.0
                                class_id = int(boxes.cls[i]) if boxes.cls is not None else 0
                                if class_id < len(self.model.names):
                                    class_name = self.model.names[class_id]
                                else:
                                    class_name = f"unknown_{class_id}"
                                if boxes.xyxy is not None:
                                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                                else:
                                    x1, y1, x2, y2 = [0, 0, 0, 0]
                                detected_objects.append({
                                    'class_name': class_name,
                                    'confidence': confidence,
                                    'bbox': [x1, y1, x2, y2],
                                    'class_id': class_id
                                })
                            except Exception as box_error:
                                logger.warning(f"Error processing box {i}: {str(box_error)}")
                                continue
            logger.info(f"Successfully detected {len(detected_objects)} objects")
            return detected_objects
        except Exception as e:
            logger.error(f"Object detection failed: {str(e)}")
            return []

class ThreatAnalyzer:
    def __init__(self, api_key: str):
        try:
            self.client = genai.Client(api_key=api_key)
            model_names = ['gemini-2.0-flash']
            self.model_name = None
            for model_name in model_names:
                try:
                    test_response = self.client.models.generate_content(
                        model=model_name,
                        contents='Test'
                    )
                    self.model_name = model_name
                    logger.info(f"Gemini LLM initialized successfully with model: {model_name}")
                    break
                except Exception as model_error:
                    logger.warning(f"Failed to initialize model {model_name}: {str(model_error)}")
                    continue
            if self.model_name is None:
                raise Exception("All Gemini models failed to initialize")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini LLM: {str(e)}")
            raise
    
    def analyze_threat(self, detected_objects: List[Dict], context: str = "", image_data: Optional[bytes] = None) -> Dict:
        try:
            object_list = [obj['class_name'] for obj in detected_objects] if detected_objects else []
            object_counts = {}
            for obj in object_list:
                object_counts[obj] = object_counts.get(obj, 0) + 1
            logger.info(f"Analyzing threat for objects: {object_counts}")
            if detected_objects:
                prompt = f"""
                You are an expert security analyst with computer vision capabilities. Analyze this scene for potential threats.
                
                YOLO DETECTED OBJECTS:
                {json.dumps(object_counts, indent=2)}
                
                CONTEXT: {context}
                
                Perform comprehensive analysis:
                1. Review the YOLO-detected objects for weapons or dangerous items
                2. Look for unusual object combinations that might indicate suspicious activity
                3. If an image is provided, analyze it visually for any threats YOLO might have missed
                4. Consider context-appropriate threat assessment
                5. Look for people, weapons, suspicious behavior, or dangerous situations
                
                Respond ONLY in valid JSON format:
                {{
                    "threat_level": "LOW",
                    "is_suspicious": false,
                    "explanation": "Your detailed analysis combining YOLO results and visual inspection",
                    "recommended_action": "Action to take based on complete analysis",
                    "confidence": 0.85,
                    "additional_objects_found": ["any additional objects you visually identified"]
                }}
                
                Use only LOW/MEDIUM/HIGH for threat_level. Be thorough but conservative.
                """
            else:
                prompt = f"""
                You are an expert security analyst with computer vision capabilities. YOLO object detection found no objects in this image, but you need to perform visual analysis to ensure nothing was missed.
                
                YOLO RESULT: No objects detected
                CONTEXT: {context}
                
                Perform comprehensive visual analysis:
                1. Carefully examine the entire image for any people, objects, or activities
                2. Look for weapons, dangerous items, or suspicious behavior
                3. Identify any security threats that automated detection might have missed
                4. Consider lighting, image quality, or other factors that might affect detection
                5. Assess overall scene safety and security implications
                
                Respond ONLY in valid JSON format:
                {{
                    "threat_level": "LOW",
                    "is_suspicious": false,
                    "explanation": "Your detailed visual analysis of what you can see in the image",
                    "recommended_action": "Action to take based on visual inspection",
                    "confidence": 0.85,
                    "objects_identified": ["list any objects or people you can identify"]
                }}
                
                Use only LOW/MEDIUM/HIGH for threat_level. Be thorough - this is the only analysis being performed.
                """
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    logger.info(f"LLM analysis attempt {attempt + 1} {'with image' if image_data else 'text-only'}")
                    if image_data:
                        content = [
                            genai_types.Part.from_text(prompt),
                            genai_types.Part.from_bytes(data=image_data, mime_type="image/jpeg"),
                        ]
                    else:
                        content = prompt
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=content,
                        config=genai_types.GenerateContentConfig(
                            temperature=0.3,
                            top_p=0.8,
                            top_k=40,
                            max_output_tokens=1000,
                        )
                    )
                    if not response or not response.text:
                        logger.warning(f"Empty response from LLM on attempt {attempt + 1}")
                        continue
                    analysis_text = response.text.strip()
                    logger.info(f"Raw LLM response: {analysis_text[:200]}...")
                    if '```json' in analysis_text:
                        start = analysis_text.find('```json') + 7
                        end = analysis_text.find('```', start)
                        analysis_text = analysis_text[start:end]
                    elif '```' in analysis_text:
                        start = analysis_text.find('```') + 3
                        end = analysis_text.find('```', start)
                        analysis_text = analysis_text[start:end]
                    analysis_text = analysis_text.strip()
                    analysis = json.loads(analysis_text)
                    required_fields = ['threat_level', 'is_suspicious', 'explanation', 'recommended_action', 'confidence']
                    if not all(field in analysis for field in required_fields):
                        logger.warning(f"Missing fields in LLM response: {analysis}")
                        continue
                    if analysis['threat_level'] not in ['LOW', 'MEDIUM', 'HIGH']:
                        analysis['threat_level'] = 'LOW'
                    if not isinstance(analysis['confidence'], (int, float)) or not 0 <= analysis['confidence'] <= 1:
                        analysis['confidence'] = 0.5
                    logger.info(f"Threat analysis completed: {analysis['threat_level']} threat level")
                    return analysis
                except json.JSONDecodeError as json_error:
                    logger.warning(f"JSON parsing failed on attempt {attempt + 1}: {str(json_error)}")
                    logger.warning(f"Raw text: {analysis_text}")
                    continue
                except Exception as attempt_error:
                    logger.warning(f"Analysis attempt {attempt + 1} failed: {str(attempt_error)}")
                    continue
            logger.error("All LLM analysis attempts failed")
            if detected_objects:
                explanation = f"Analysis of {len(object_counts)} object types completed, but LLM response parsing failed. Objects detected: {', '.join(object_counts.keys())}. Manual review recommended."
            else:
                explanation = "No objects detected by YOLO and LLM visual analysis failed. Manual review strongly recommended to ensure scene safety."
            return {
                "threat_level": "LOW" if detected_objects else "MEDIUM",
                "is_suspicious": False,
                "explanation": explanation,
                "recommended_action": "Manual review of the scene is recommended due to technical issues.",
                "confidence": 0.3
            }
        except Exception as e:
            logger.error(f"Threat analysis failed completely: {str(e)}")
            return {
                "threat_level": "UNKNOWN",
                "is_suspicious": False,
                "explanation": f"Threat analysis failed due to technical error: {str(e)}",
                "recommended_action": "Technical review required.",
                "confidence": 0.0
            }

class AlertAgent:
    def __init__(self):
        self.alert_log = []
    
    def send_authority_alert(self, analysis: Dict, detected_objects: List[Dict]) -> bool:
        try:
            alert_message = {
                "timestamp": datetime.now().isoformat(),
                "threat_level": analysis["threat_level"],
                "location": "Camera Feed",
                "detected_objects": [obj['class_name'] for obj in detected_objects],
                "analysis": analysis["explanation"],
                "recommended_action": analysis["recommended_action"]
            }
            logger.warning(f"SECURITY ALERT: {json.dumps(alert_message, indent=2)}")
            self.alert_log.append(alert_message)
            return True
        except Exception as e:
            logger.error(f"Failed to send authority alert: {str(e)}")
            return False
    
    def generate_user_message(self, analysis: Dict, detected_objects: List[Dict], alert_sent: bool = False, had_yolo_detection: bool = True) -> str:
        try:
            config = ConfigManager()
            if not config.gemini_api_key:
                return "System message: Analysis completed. Please check the detailed results above."
            client = genai.Client(api_key=config.gemini_api_key)
            model_name = 'gemini-2.0-flash-lite'
            detection_method = "YOLO object detection and AI vision analysis" if had_yolo_detection else "comprehensive AI vision analysis (YOLO detected no objects)"
            prompt = f"""
            Generate a clear, professional message for a user of a security monitoring system.
            
            ANALYSIS RESULTS:
            - Detection Method Used: {detection_method}
            - Threat Level: {analysis['threat_level']}
            - Suspicious Activity: {analysis['is_suspicious']}
            - Analysis: {analysis['explanation']}
            - Confidence: {analysis['confidence']}
            - Alert Sent to Authorities: {alert_sent}
            
            DETECTED OBJECTS: {[obj['class_name'] for obj in detected_objects] if detected_objects else "None by YOLO"}
            
            Create a concise, professional message that:
            1. Explains the detection method used (dual-layer vs vision-only analysis)
            2. Summarizes the analysis results in simple terms
            3. Indicates what action was taken (if any)
            4. Provides appropriate reassurance or caution based on the threat level
            5. Mentions the thoroughness of the analysis system
            
            Keep it under 150 words and maintain a professional but friendly tone.
            """
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Failed to generate user message: {str(e)}")
            detection_info = "dual-layer analysis" if had_yolo_detection else "AI vision analysis"
            return f"Analysis completed using {detection_info} with {analysis['threat_level']} threat level. " + \
                   ("Authorities have been notified." if alert_sent else "No immediate action required.")

class ThreatDetectionApp:
    def __init__(self):
        self.config = ConfigManager()
        self.detector = None
        self.analyzer = None
        self.agent = AlertAgent()
        self.initialization_error = None
        
    def initialize_components(self):
        """Initialize components with current API key configuration"""
        is_valid, api_key = self.config.validate_config()
        if is_valid:
            try:
                if self.detector is None:
                    logger.info("Initializing YOLO detector...")
                    self.detector = ObjectDetector(self.config.model_path)
                    logger.info("YOLO detector initialized successfully")
                
                if self.analyzer is None:
                    logger.info("Initializing Gemini analyzer...")
                    self.analyzer = ThreatAnalyzer(api_key)
                    logger.info("Gemini analyzer initialized successfully")
                
                self.initialization_error = None
                return True
                
            except Exception as e:
                error_msg = f"Failed to initialize components: {str(e)}"
                self.initialization_error = error_msg
                logger.error(f"Component initialization failed: {str(e)}")
                return False
        else:
            self.initialization_error = "Configuration validation failed"
            return False
    
    def process_image(self, uploaded_file) -> Tuple[List[Dict], Dict, str]:
        try:
            if self.detector is None:
                raise Exception("Object detector not initialized. Check YOLO model loading.")
            if self.analyzer is None:
                raise Exception("Threat analyzer not initialized. Check Gemini API configuration.")
            st.info("📷 Loading and preprocessing image...")
            image = Image.open(uploaded_file)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image_np = np.array(image)
            logger.info(f"Image loaded: {image_np.shape}, dtype: {image_np.dtype}")
            if image_np is None or image_np.size == 0:
                raise Exception("Invalid image data")
            image_bytes = None
            try:
                uploaded_file.seek(0)
                image_bytes = uploaded_file.read()
                if not uploaded_file.name.lower().endswith(('.jpg', '.jpeg')):
                    buffer = io.BytesIO()
                    image.save(buffer, format='JPEG', quality=85)
                    image_bytes = buffer.getvalue()
                    buffer.close()
                logger.info(f"Image prepared for LLM analysis: {len(image_bytes)} bytes")
                uploaded_file.seek(0)
            except Exception as img_prep_error:
                logger.warning(f"Failed to prepare image for LLM: {str(img_prep_error)}")
                image_bytes = None
            st.info("🔍 Detecting objects in the image...")
            detected_objects = self.detector.detect_objects(image_np)
            if detected_objects:
                object_names = [obj['class_name'] for obj in detected_objects]
                logger.info(f"Objects detected by YOLO: {object_names}")
                st.success(f"✅ YOLO detected {len(detected_objects)} objects")
            else:
                logger.info("No objects detected by YOLO - will rely on LLM vision analysis")
                st.warning("⚠️ YOLO detected no objects - performing AI vision analysis...")
            st.info("🧠 Analyzing potential threats with AI vision...")
            analysis = self.analyzer.analyze_threat(detected_objects, context="Security monitoring system", image_data=image_bytes)
            if not analysis or 'threat_level' not in analysis:
                logger.error("Invalid analysis result from LLM")
                analysis = {
                    "threat_level": "MEDIUM",
                    "is_suspicious": False,
                    "explanation": "Analysis failed to produce valid results. Manual review strongly recommended.",
                    "recommended_action": "Manual review recommended.",
                    "confidence": 0.3
                }
            if not detected_objects:
                additional_info = ""
                if 'objects_identified' in analysis:
                    additional_info = f" LLM identified: {analysis['objects_identified']}"
                elif 'additional_objects_found' in analysis:
                    additional_info = f" LLM found: {analysis['additional_objects_found']}"
                logger.info(f"No YOLO detections, LLM analysis: {analysis['threat_level']} threat.{additional_info}")
                no_objects_message = f"""
                YOLO object detection found no objects in this image, but our AI vision system has performed 
                a comprehensive visual analysis. 
                
                **Analysis Result**: {analysis['threat_level']} threat level
                **AI Assessment**: {analysis['explanation']}
                
                This dual-layer approach ensures thorough security monitoring even when automated 
                object detection doesn't identify specific items.
                """
                return [], analysis, no_objects_message.strip()
            alert_sent = False
            if analysis.get("is_suspicious", False) and analysis.get("threat_level", "LOW") in ["MEDIUM", "HIGH"]:
                st.warning("⚠️ Suspicious activity detected. Notifying authorities...")
                alert_sent = self.agent.send_authority_alert(analysis, detected_objects)
            st.info("💬 Generating summary message...")
            user_message = self.agent.generate_user_message(analysis, detected_objects, alert_sent, had_yolo_detection=bool(detected_objects))
            logger.info("Image processing completed successfully")
            return detected_objects, analysis, user_message
        except Exception as e:
            logger.error(f"Image processing failed: {str(e)}")
            st.error(f"Processing Error: {str(e)}")
            error_analysis = {
                "threat_level": "ERROR",
                "is_suspicious": False,
                "explanation": f"Processing failed: {str(e)}",
                "recommended_action": "Please try again with a different image or check system configuration.",
                "confidence": 0.0
            }
            return [], error_analysis, f"Sorry, there was an error processing your image: {str(e)}"
    
    def run(self):
        st.set_page_config(
            page_title="AI Threat Detection System",
            page_icon="🛡️",
            layout="wide"
        )
        
        st.title("🛡️ AI-Powered Threat Detection System")
        st.markdown("Upload an image to analyze for potential security threats using **dual-layer AI detection**: YOLO object detection + Gemini Vision analysis for comprehensive threat assessment.")
        
        with st.expander("🔍 How it works"):
            st.markdown("""
            **Dual-Layer Detection Process:**
            
            1. **YOLO Object Detection**: Identifies specific objects, people, and items with confidence scores
            2. **AI Vision Analysis**: Google Gemini examines the entire image for threats that might be missed
            3. **Combined Assessment**: Both analyses are merged for comprehensive threat evaluation
            4. **Smart Fallback**: When YOLO finds nothing, AI Vision performs complete visual inspection
            
            This ensures maximum detection accuracy and minimizes false negatives.
            """)
        
        with st.sidebar:
            st.header("ℹ️ About")
            st.markdown("""
            This system uses:
            - **YOLOv8** for object detection
            - **Google Gemini Vision** for AI image analysis
            - **Dual-layer detection** for comprehensive coverage
            - **Automated alerts** for suspicious activities
            
            **Detection Methods:**
            1. **YOLO** identifies specific objects with confidence scores
            2. **AI Vision** performs visual analysis when YOLO detection is insufficient
            3. **Combined Analysis** ensures nothing is missed
            
            **Supported formats:** JPG, JPEG, PNG
            """)
        
        # Initialize components (this handles API key configuration in sidebar)
        components_ready = self.initialize_components()
        
        with st.sidebar:
            st.header("🔧 System Status")
            is_valid, _ = self.config.validate_config()
            
            if is_valid and components_ready:
                st.success("✅ System Ready")
                st.info(f"🤖 YOLO Model: {self.config.model_path}")
                st.info("🧠 LLM: Google Gemini Vision")
                st.info("🔍 Dual-Layer Detection: Active")
            elif is_valid and not components_ready:
                st.error(f"❌ System Error: {self.initialization_error}")
                st.info("Please check your API key and try again.")
            else:
                st.error("❌ Configuration Error")
                st.info("Please provide a valid Gemini API key to continue.")
        
        # Only show file uploader if components are ready
        if components_ready:
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['jpg', 'jpeg', 'png'],
                help="Upload an image to analyze for potential threats"
            )
            
            if uploaded_file is not None:
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.subheader("📷 Uploaded Image")
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                with col2:
                    st.subheader("🔄 Processing Status")
                    with st.spinner("Processing image..."):
                        detected_objects, analysis, user_message = self.process_image(uploaded_file)
                    st.success("✅ Analysis Complete!")
                
                st.markdown("---")
                st.header("📊 Analysis Results")
                
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    st.subheader("🎯 Object Detection Results")
                    if detected_objects:
                        st.write("**YOLO Detected Objects:**")
                        for obj in detected_objects:
                            confidence_pct = obj['confidence'] * 100
                            st.write(f"• **{obj['class_name']}** ({confidence_pct:.1f}%)")
                    else:
                        st.write("**YOLO Detection:** No objects found")
                    
                    if 'objects_identified' in analysis and analysis['objects_identified']:
                        st.write("**AI Vision Identified:**")
                        for obj in analysis['objects_identified']:
                            st.write(f"• **{obj}** (Vision AI)")
                    elif 'additional_objects_found' in analysis and analysis['additional_objects_found']:
                        st.write("**Additional Objects Found by AI:**")
                        for obj in analysis['additional_objects_found']:
                            st.write(f"• **{obj}** (Vision AI)")
                    
                    if not detected_objects and not analysis.get('objects_identified') and not analysis.get('additional_objects_found'):
                        st.info("🔍 Both YOLO and AI Vision performed comprehensive analysis")
                
                with col2:
                    st.subheader("🧠 AI Analysis")
                    threat_level = analysis.get('threat_level', 'UNKNOWN')
                    if threat_level == 'HIGH':
                        st.error(f"🚨 **Threat Level:** {threat_level}")
                    elif threat_level == 'MEDIUM':
                        st.warning(f"⚠️ **Threat Level:** {threat_level}")
                    else:
                        st.success(f"✅ **Threat Level:** {threat_level}")
                    
                    is_suspicious = analysis.get('is_suspicious', False)
                    if is_suspicious:
                        st.error("🔍 **Suspicious Activity:** Yes")
                    else:
                        st.success("🔍 **Suspicious Activity:** No")
                    
                    confidence = analysis.get('confidence', 0.0)
                    st.metric("📈 Confidence", f"{confidence:.1%}")
                
                with col3:
                    st.subheader("💬 AI Reasoning")
                    explanation = analysis.get('explanation', 'No explanation available')
                    st.write(explanation)
                    recommended_action = analysis.get('recommended_action', 'None')
                    st.write(f"**Recommended Action:** {recommended_action}")
                
                st.markdown("---")
                st.header("📋 Summary")
                st.info(user_message)
                
                if hasattr(self.agent, 'alert_log') and self.agent.alert_log:
                    with st.expander("🚨 Alert Log"):
                        for alert in self.agent.alert_log:
                            st.json(alert)
        else:
            # Show helpful message when system is not ready
            st.info("🔧 **System Configuration Required**")
            st.markdown("""
            To use the threat detection system, please:
            1. Enter your Google Gemini API key in the sidebar
            2. Wait for system validation
            3. Upload an image for analysis
            
            Get your API key from: [Google AI Studio](https://makersuite.google.com/app/apikey)
            """)

def main():
    try:
        app = ThreatDetectionApp()
        app.run()
    except Exception as e:
        st.error(f"Application failed to start: {str(e)}")
        logger.error(f"Application startup failed: {str(e)}")

if __name__ == "__main__":
    main()

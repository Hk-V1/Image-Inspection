import streamlit as st
import sqlite3
import google.generativeai as genai
from PIL import Image
import io
import base64
import json
import datetime
import pandas as pd
import os
from typing import Dict, Any, Optional

class Config:
    DATABASE_PATH = "rice_quality.db"
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 
    
    QUALITY_CRITERIA = {
        "grain_length": {"min": 5.0, "max": 7.0, "unit": "mm"},
        "grain_width": {"min": 2.0, "max": 3.5, "unit": "mm"},
        "broken_grains": {"max": 5, "unit": "%"},
        "discolored_grains": {"max": 2, "unit": "%"},
        "foreign_matter": {"max": 0.1, "unit": "%"},
        "moisture_content": {"min": 12, "max": 14, "unit": "%"},
        "overall_grade": ["A", "B", "C", "D", "F"]
    }

class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS rice_evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                image_name TEXT NOT NULL,
                image_data BLOB,
                evaluation_result TEXT NOT NULL,
                grain_length REAL,
                grain_width REAL,
                broken_grains REAL,
                discolored_grains REAL,
                foreign_matter REAL,
                moisture_content REAL,
                overall_grade TEXT,
                quality_score REAL,
                passed_quality_check BOOLEAN,
                notes TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_evaluation(self, data: Dict[str, Any]) -> int:
        """Save evaluation result to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO rice_evaluations 
            (image_name, image_data, evaluation_result, grain_length, grain_width, 
             broken_grains, discolored_grains, foreign_matter, moisture_content, 
             overall_grade, quality_score, passed_quality_check, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data['image_name'],
            data['image_data'],
            data['evaluation_result'],
            data.get('grain_length'),
            data.get('grain_width'),
            data.get('broken_grains'),
            data.get('discolored_grains'),
            data.get('foreign_matter'),
            data.get('moisture_content'),
            data.get('overall_grade'),
            data.get('quality_score'),
            data.get('passed_quality_check'),
            data.get('notes')
        ))
        
        evaluation_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return evaluation_id
    
    def get_evaluations(self, limit: int = 100) -> pd.DataFrame:
        """Retrieve evaluation history"""
        conn = sqlite3.connect(self.db_path)
        query = '''
            SELECT id, timestamp, image_name, grain_length, grain_width, 
                   broken_grains, discolored_grains, foreign_matter, 
                   moisture_content, overall_grade, quality_score, 
                   passed_quality_check, notes
            FROM rice_evaluations 
            ORDER BY timestamp DESC 
            LIMIT ?
        '''
        df = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()
        return df
    
    def get_evaluation_by_id(self, eval_id: int) -> Optional[Dict]:
        """Get specific evaluation by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM rice_evaluations WHERE id = ?
        ''', (eval_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, result))
        return None

class GeminiEvaluator:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
    
    def evaluate_rice_image(self, image: Image.Image) -> Dict[str, Any]:
        """Evaluate rice grain image using Gemini API"""
        
        # Quality control prompt for dataset evaluation
        prompt = """
        You are evaluating this image of rice grains for quality control in the context of creating a dataset for AI/ML image prediction and classification (e.g., for grain counting, classification, or defect detection). Please assess the image based on the following criteria, keeping in mind the needs of machine learning algorithms:

        A. Image Clarity, Focus, and Lighting:
        Is the image sharp and free from any blur (motion or out-of-focus)? (yes/no)
        Is the lighting even and sufficient across the entire image, allowing for clear distinction of grain features (e.g., no harsh shadows, overexposure, or underexposed areas that obscure details)? (yes/no)

        B. Background Uniformity and Object Isolation:
        Are all visible rice grains entirely located on a monochromatic (single color, e.g., black) and truly uniform background (i.e., no patterns, textures, or varying shades that could be confused for grain features)? (yes/no)
        Are there absolutely no foreign objects, debris, dust particles, other types of grains, or any other materials present on the background or among the rice grains? (yes/no)
        Do the grains occupy a significant and appropriate portion of the image frame, allowing for clear detail without being excessively zoomed in (cutting off grains) or zoomed out (grains too small for detailed analysis)? (yes/no)

        C. Grain Visibility, Separation, and Integrity:
        Are all individual rice grains distinctly visible, free from any partial obscuring (e.g., by shadows, reflections, or other grains)? (yes/no)
        Are all individual rice grains sufficiently separated from one another, allowing for clear segmentation and individual analysis by an algorithm (i.e., no significant overlapping or touching that would make it difficult to identify distinct grains)? (yes/no)
        Are all grains fully visible within the image frame, with no partial grains cut off at the edges or cropped out? (yes/no)

        Overall Verdict: Does this image meet ALL the specified quality standards for inclusion in a high-quality AI/ML dataset for rice grain analysis? (Final answer: yes or no)

        Respond in JSON:
        {
            "sharp": "yes/no",
            "lighting": "yes/no",
            "background": "yes/no",
            "clean": "yes/no",
            "size": "yes/no",
            "visible": "yes/no",
            "separated": "yes/no",
            "complete": "yes/no",
            "verdict": "yes/no",
            "issues": "brief explanation if rejected"
        }

        Provide only the JSON response, no additional text before or after.
        """
        
        try:
            response = self.model.generate_content([prompt, image])
            response_text = response.text.strip()
            
            # Debug: Print raw response for troubleshooting
            print(f"Raw API Response: {response_text}")
            
            # Clean the response text
            response_text = response_text.replace('```json', '').replace('```', '').strip()
            
            # Try to find and extract JSON
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                try:
                    evaluation_data = json.loads(json_str)
                    
                    # Validate required fields and provide defaults
                    required_fields = {
                        'sharp': 'yes',
                        'lighting': 'yes',
                        'background': 'yes',
                        'clean': 'yes',
                        'size': 'yes',
                        'visible': 'yes',
                        'separated': 'yes',
                        'complete': 'yes',
                        'verdict': 'yes',
                        'issues': 'No issues detected'
                    }
                    
                    # Fill in missing fields
                    for field, default_value in required_fields.items():
                        if field not in evaluation_data or evaluation_data[field] is None:
                            evaluation_data[field] = default_value
                    
                    # Calculate overall quality score based on criteria
                    score = 0
                    criteria_count = 8  # Number of yes/no criteria
                    for criterion in ['sharp', 'lighting', 'background', 'clean', 'size', 'visible', 'separated', 'complete']:
                        if evaluation_data.get(criterion, 'no').lower() == 'yes':
                            score += 1
                    
                    quality_score = (score / criteria_count) * 100
                    passed_check = evaluation_data.get('verdict', 'no').lower() == 'yes'
                    
                    # Add computed fields for compatibility with existing system
                    evaluation_data.update({
                        'quality_score': quality_score,
                        'passed_quality_check': passed_check,
                        'overall_grade': 'A' if score >= 7 else 'B' if score >= 6 else 'C' if score >= 5 else 'D' if score >= 4 else 'F',
                        'detailed_analysis': self._generate_analysis_from_criteria(evaluation_data),
                        'recommendations': self._generate_recommendations_from_criteria(evaluation_data)
                    })
                    
                    # Add raw response for debugging
                    evaluation_data['raw_response'] = response_text
                    
                    return evaluation_data
                    
                except json.JSONDecodeError as je:
                    print(f"JSON parsing error: {je}")
                    return self._get_fallback_evaluation(response_text)
            else:
                print("No JSON found in response")
                return self._get_fallback_evaluation(response_text)
                
        except Exception as e:
            print(f"API call failed: {str(e)}")
            return {
                "error": f"Evaluation failed: {str(e)}",
                "grain_length": 0,
                "grain_width": 0,
                "broken_grains": 0,
                "discolored_grains": 0,
                "foreign_matter": 0,
                "moisture_content": 0,
                "overall_grade": "F",
                "quality_score": 0,
                "passed_quality_check": False,
                "detailed_analysis": f"Error occurred during evaluation: {str(e)}",
                "recommendations": "Please try again or check your API configuration."
            }
    
    def _get_fallback_evaluation(self, response_text: str) -> Dict[str, Any]:
        """Provide fallback evaluation when JSON parsing fails"""
        return {
            'sharp': 'yes',
            'lighting': 'yes',
            'background': 'yes',
            'clean': 'yes',
            'size': 'yes',
            'visible': 'yes',
            'separated': 'yes',
            'complete': 'yes',
            'verdict': 'yes',
            'issues': 'Fallback evaluation - manual review needed',
            'quality_score': 75,
            'passed_quality_check': True,
            'overall_grade': 'C',
            'detailed_analysis': 'Fallback evaluation used due to API response parsing issues.',
            'recommendations': 'Manual inspection recommended for accurate results.',
            'raw_response': response_text,
            'note': 'This is a fallback evaluation due to API response parsing issues.'
        }
    
    def _generate_analysis_from_criteria(self, data: Dict[str, Any]) -> str:
        """Generate detailed analysis based on evaluation criteria"""
        analysis = "Dataset Quality Assessment:\n"
        
        criteria_map = {
            'sharp': 'Image sharpness and focus',
            'lighting': 'Lighting quality and uniformity',
            'background': 'Background uniformity',
            'clean': 'Absence of foreign objects',
            'size': 'Appropriate grain sizing in frame',
            'visible': 'Grain visibility',
            'separated': 'Grain separation for segmentation',
            'complete': 'Complete grains within frame'
        }
        
        passed = []
        failed = []
        
        for criterion, description in criteria_map.items():
            if data.get(criterion, 'no').lower() == 'yes':
                passed.append(description)
            else:
                failed.append(description)
        
        if passed:
            analysis += f"Passed criteria: {', '.join(passed)}\n"
        
        if failed:
            analysis += f"Failed criteria: {', '.join(failed)}\n"
        
        if data.get('issues'):
            analysis += f"Issues noted: {data['issues']}"
        
        return analysis
    
    def _generate_recommendations_from_criteria(self, data: Dict[str, Any]) -> str:
        """Generate recommendations based on failed criteria"""
        recommendations = []
        
        if data.get('sharp', 'yes').lower() == 'no':
            recommendations.append("Improve image sharpness by using better focus and reducing camera shake")
        
        if data.get('lighting', 'yes').lower() == 'no':
            recommendations.append("Ensure even lighting across the entire image to avoid shadows and overexposure")
        
        if data.get('background', 'yes').lower() == 'no':
            recommendations.append("Use a uniform, monochromatic background for better grain isolation")
        
        if data.get('clean', 'yes').lower() == 'no':
            recommendations.append("Remove all foreign objects and debris from the imaging area")
        
        if data.get('size', 'yes').lower() == 'no':
            recommendations.append("Adjust camera distance to ensure grains occupy appropriate frame portion")
        
        if data.get('visible', 'yes').lower() == 'no':
            recommendations.append("Ensure all grains are clearly visible without shadows or reflections")
        
        if data.get('separated', 'yes').lower() == 'no':
            recommendations.append("Spread grains further apart to enable proper algorithmic segmentation")
        
        if data.get('complete', 'yes').lower() == 'no':
            recommendations.append("Ensure all grains are fully within the image frame")
        
        if not recommendations:
            return "Image meets all quality standards for dataset inclusion."
        
        return "Recommendations for improvement:\n• " + "\n• ".join(recommendations)

# Streamlit UI
def main():
    st.set_page_config(
        page_title="Rice Grain Quality Control System",
        page_icon="🌾",
        layout="wide"
    )
    
    st.title("Rice Grain Quality Control System")
    st.markdown("Upload an image of rice grains for quality evaluation")
    
    # Initialize components
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager(Config.DATABASE_PATH)
    
    if 'gemini_evaluator' not in st.session_state:
        if Config.GEMINI_API_KEY:
            st.session_state.gemini_evaluator = GeminiEvaluator(Config.GEMINI_API_KEY)
        else:
            st.error("Please set your GEMINI_API_KEY environment variable")
            st.stop()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Upload & Evaluate", "View History", "Quality Standards"])
    
    if page == "Upload & Evaluate":
        upload_and_evaluate_page()
    elif page == "View History":
        view_history_page()
    elif page == "Quality Standards":
        quality_standards_page()

def upload_and_evaluate_page():
    st.header("Upload Rice Grain Image")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload a clear image of rice grains"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Uploaded Image")
            st.image(image, caption="Rice grains to evaluate", use_column_width=True)
        
        with col2:
            st.subheader("Evaluation")
            
            if st.button("🔍 Evaluate Quality", type="primary"):
                with st.spinner("Evaluating rice grain quality..."):
                    
                    # Check API key first
                    if not Config.GEMINI_API_KEY:
                        st.error("Gemini API key not found. Please set your GEMINI_API_KEY environment variable.")
                        return
                    
                    try:
                        # Evaluate using Gemini
                        evaluation_result = st.session_state.gemini_evaluator.evaluate_rice_image(image)
                        
                        # Convert image to binary for storage
                        img_byte_arr = io.BytesIO()
                        image.save(img_byte_arr, format='PNG')
                        img_binary = img_byte_arr.getvalue()
                        
                        # Prepare data for database
                        db_data = {
                            'image_name': uploaded_file.name,
                            'image_data': img_binary,
                            'evaluation_result': json.dumps(evaluation_result),
                            'grain_length': None,  # Not used in new prompt
                            'grain_width': None,   # Not used in new prompt
                            'broken_grains': None, # Not used in new prompt
                            'discolored_grains': None, # Not used in new prompt
                            'foreign_matter': None, # Not used in new prompt
                            'moisture_content': None, # Not used in new prompt
                            'overall_grade': evaluation_result.get('overall_grade'),
                            'quality_score': evaluation_result.get('quality_score'),
                            'passed_quality_check': evaluation_result.get('passed_quality_check'),
                            'notes': evaluation_result.get('detailed_analysis', '')
                        }
                        
                        # Save to database
                        eval_id = st.session_state.db_manager.save_evaluation(db_data)
                        
                        # Display results
                        display_evaluation_results(evaluation_result, eval_id)
                        
                    except Exception as e:
                        st.error(f"Error during evaluation: {str(e)}")
                        st.error("Please check your API key and internet connection.")

def display_evaluation_results(evaluation_result: Dict[str, Any], eval_id: int):
    """Display evaluation results in a formatted way"""
    
    if 'error' in evaluation_result:
        st.error(f"Evaluation Error: {evaluation_result['error']}")
        if 'raw_response' in evaluation_result:
            with st.expander("Raw API Response (for debugging)"):
                st.text(evaluation_result['raw_response'])
        return
    
    # Overall result
    passed = evaluation_result.get('passed_quality_check', False)
    quality_score = evaluation_result.get('quality_score', 0)
    overall_grade = evaluation_result.get('overall_grade', 'N/A')
    verdict = evaluation_result.get('verdict', 'no')
    
    if passed and verdict.lower() == 'yes':
        st.success(f"Dataset Quality Check: PASSED (Score: {quality_score:.1f}/100, Grade: {overall_grade})")
    else:
        st.error(f"Dataset Quality Check: FAILED (Score: {quality_score:.1f}/100, Grade: {overall_grade})")
    
    # Dataset quality criteria
    st.subheader("Dataset Quality Criteria")
    
    criteria_labels = {
        'sharp': 'Image Sharpness & Focus',
        'lighting': 'Lighting Quality',
        'background': 'Background Uniformity',
        'clean': 'Clean (No Foreign Objects)',
        'size': 'Appropriate Grain Size',
        'visible': 'Grain Visibility',
        'separated': 'Grain Separation',
        'complete': 'Complete Grains in Frame'
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        for i, (criterion, label) in enumerate(list(criteria_labels.items())[:4]):
            value = evaluation_result.get(criterion, 'no')
            if value.lower() == 'yes':
                st.success(f"{label}: PASS")
            else:
                st.error(f"{label}: FAIL")
    
    with col2:
        for i, (criterion, label) in enumerate(list(criteria_labels.items())[4:]):
            value = evaluation_result.get(criterion, 'no')
            if value.lower() == 'yes':
                st.success(f"{label}: PASS")
            else:
                st.error(f"{label}: FAIL")
    
    # Issues if any
    if 'issues' in evaluation_result and evaluation_result['issues']:
        st.subheader("Issues Identified")
        st.warning(evaluation_result['issues'])
    
    # Detailed analysis
    if 'detailed_analysis' in evaluation_result and evaluation_result['detailed_analysis']:
        st.subheader("Detailed Analysis")
        st.write(evaluation_result['detailed_analysis'])
    
    # Recommendations
    if 'recommendations' in evaluation_result and evaluation_result['recommendations']:
        st.subheader("Recommendations")
        st.write(evaluation_result['recommendations'])
    
    # Debug info
    if 'note' in evaluation_result:
        st.warning(f"Note: {evaluation_result['note']}")
    
    # Raw response for debugging (expandable)
    if 'raw_response' in evaluation_result:
        with st.expander("Raw API Response (for debugging)"):
            st.text(evaluation_result['raw_response'])
    
    st.info(f"Evaluation saved with ID: {eval_id}")

def view_history_page():
    st.header("Evaluation History")
    
    # Get evaluation history
    df = st.session_state.db_manager.get_evaluations()
    
    if df.empty:
        st.info("No evaluations found. Upload and evaluate some rice grain images first.")
        return
    
    # Display summary statistics
    st.subheader("Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Evaluations", len(df))
    
    with col2:
        passed_count = df['passed_quality_check'].sum()
        st.metric("Passed Quality Check", passed_count)
    
    with col3:
        avg_score = df['quality_score'].mean()
        st.metric("Average Quality Score", f"{avg_score:.1f}")
    
    with col4:
        if not df.empty:
            grade_counts = df['overall_grade'].value_counts()
            most_common_grade = grade_counts.index[0] if len(grade_counts) > 0 else "N/A"
            st.metric("Most Common Grade", most_common_grade)
    
    # Display evaluation table
    st.subheader("Evaluation Records")
    
    # Format the dataframe for display
    display_df = df.copy()
    display_df['timestamp'] = pd.to_datetime(display_df['timestamp'])
    display_df['passed_quality_check'] = display_df['passed_quality_check'].map({True: '✅', False: '❌'})
    
    st.dataframe(
        display_df,
        column_config={
            "timestamp": st.column_config.DatetimeColumn("Date & Time"),
            "image_name": "Image Name",
            "quality_score": st.column_config.NumberColumn("Quality Score", format="%.1f"),
            "overall_grade": "Grade",
            "passed_quality_check": "Status"
        },
        use_container_width=True
    )

def quality_standards_page():
    st.header("Quality Standards & Criteria")
    
    st.markdown("""
    This system evaluates rice grain quality based on the following industry standards:
    """)
    
    # Display quality criteria
    criteria = Config.QUALITY_CRITERIA
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Physical Measurements")
        st.write(f"**Grain Length**: {criteria['grain_length']['min']}-{criteria['grain_length']['max']} {criteria['grain_length']['unit']}")
        st.write(f"**Grain Width**: {criteria['grain_width']['min']}-{criteria['grain_width']['max']} {criteria['grain_width']['unit']}")
        st.write(f"**Moisture Content**: {criteria['moisture_content']['min']}-{criteria['moisture_content']['max']} {criteria['moisture_content']['unit']}")
    
    with col2:
        st.subheader("Quality Defects (Max Acceptable)")
        st.write(f"**Broken Grains**: ≤{criteria['broken_grains']['max']}{criteria['broken_grains']['unit']}")
        st.write(f"**Discolored Grains**: ≤{criteria['discolored_grains']['max']}{criteria['discolored_grains']['unit']}")
        st.write(f"**Foreign Matter**: ≤{criteria['foreign_matter']['max']}{criteria['foreign_matter']['unit']}")
    
    st.subheader("Grading Scale")
    st.write("**A**: Excellent - Meets all criteria with minimal defects")
    st.write("**B**: Good - Minor deviations from ideal standards")
    st.write("**C**: Average - Acceptable quality with some defects")
    st.write("**D**: Poor - Below standards but still usable")
    st.write("**F**: Fail - Unacceptable quality")
    
    st.subheader("Image Requirements")
    st.write("""
    For best evaluation results, ensure your rice grain images have:
    - Clear, well-lit photography
    - Grains spread out (not overlapping)
    - White or neutral background
    - High resolution (at least 1024x1024 pixels)
    - Representative sample of the batch
    """)

if __name__ == "__main__":
    main()

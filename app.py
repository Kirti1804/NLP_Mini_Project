import streamlit as st
import pandas as pd
import base64, random, time, datetime, re, sqlite3, os
from streamlit_tags import st_tags
from PIL import Image
import pdfplumber
import nltk
from nltk.corpus import stopwords
import plotly.express as px
import plotly.graph_objects as go
from courses import ds_course, web_course, android_course, ios_course, uiux_course, resume_videos, interview_videos
import google.generativeai as genai
import json

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Initialize Gemini
GEMINI_API_KEY = "API KEY" 
try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash')
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

class Database:
    def __init__(self, db_name='resume_analysis.db'):
        self.conn = sqlite3.connect(db_name)
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_data'")
        if not cursor.fetchone():
            cursor.execute('''CREATE TABLE user_data (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, 
            email TEXT NOT NULL, resume_score TEXT NOT NULL, timestamp TEXT NOT NULL, page_no TEXT NOT NULL,
            predicted_field TEXT NOT NULL, user_level TEXT NOT NULL, actual_skills TEXT NOT NULL, 
            recommended_skills TEXT NOT NULL, recommended_courses TEXT NOT NULL, ats_score TEXT)''')
        self.conn.commit()
    
    def insert_data(self, data):
        cursor = self.conn.cursor()
        columns = ["name", "email", "resume_score", "timestamp", "page_no", "predicted_field", "user_level", 
                  "actual_skills", "recommended_skills", "recommended_courses"]
        if len(data) == 11: columns.append("ats_score")
        placeholders = ", ".join(["?"] * len(data))
        cursor.execute(f"INSERT INTO user_data ({', '.join(columns)}) VALUES ({placeholders})", data)
        self.conn.commit()
        
    def fetch_all_data(self): return pd.read_sql_query('SELECT * FROM user_data', self.conn)
    def close(self): self.conn.close()

class ResumeParser:
    @staticmethod
    def pdf_reader(file):
        with pdfplumber.open(file) as pdf:
            return "".join(page.extract_text() or "" for page in pdf.pages)

    @staticmethod
    def parse_resume(file_path):
        text = ResumeParser.pdf_reader(file_path)
        with pdfplumber.open(file_path) as pdf:
            no_of_pages = len(pdf.pages)
        
        gemini_extraction = GeminiProcessor.extract_resume_info(text) if GEMINI_AVAILABLE else None
        
        if gemini_extraction:
            return {
                'name': gemini_extraction.get('name', text.split('\n')[0].strip() if text.split('\n') else "Unknown"),
                'email': gemini_extraction.get('email', ResumeParser._extract_email(text)),
                'mobile_number': gemini_extraction.get('phone', ResumeParser._extract_phone(text)),
                'skills': gemini_extraction.get('skills', ResumeParser._extract_skills(text)),
                'no_of_pages': no_of_pages,
                'text': text
            }
        else:
            return {
                'name': text.split('\n')[0].strip() if text.split('\n') else "Unknown",
                'email': ResumeParser._extract_email(text),
                'mobile_number': ResumeParser._extract_phone(text),
                'skills': ResumeParser._extract_skills(text),
                'no_of_pages': no_of_pages,
                'text': text
            }
    
    @staticmethod
    def _extract_email(text):
        email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        return email_match.group() if email_match else "No email found"
    
    @staticmethod
    def _extract_phone(text):
        phone_match = re.search(r'(\+\d{1,3})?\s*\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})', text)
        return phone_match.group() if phone_match else "No phone found"
    
    @staticmethod
    def _extract_skills(text):
        common_skills = ['python', 'java', 'c++', 'c#', 'react', 'javascript', 'angular', 'node.js', 'html', 
                        'css', 'sql', 'nosql', 'mongodb', 'postgresql', 'mysql', 'oracle', 'docker', 
                        'kubernetes', 'aws', 'azure', 'gcp', 'machine learning', 'ai', 'deep learning', 
                        'data science', 'tensorflow', 'pytorch', 'keras', 'pandas', 'numpy', 'scikit-learn', 
                        'flask', 'django', 'spring', 'hibernate', 'git', 'devops', 'ci/cd', 'jenkins', 
                        'agile', 'scrum', 'product management', 'android', 'ios', 'swift', 'kotlin', 
                        'flutter', 'react native', 'ux', 'ui', 'figma', 'adobe xd', 'photoshop', 'illustrator']
        return [skill for skill in common_skills if skill in text.lower()]

class GeminiProcessor:
    @staticmethod
    def extract_resume_info(text):
        try:
            if not GEMINI_AVAILABLE: return None
            
            prompt = f"""
            Extract the following information from this resume text:
            1. Full name
            2. Email address
            3. Phone number
            4. Skills (technical and soft skills)
            
            Return the information in JSON format with these keys: "name", "email", "phone", "skills"
            Skills should be an array of strings.
            
            Resume text:
            {text[:10000]}
            """
            
            response = model.generate_content(prompt)
            
            try:
                return json.loads(response.text)
            except:
                json_match = re.search(r'\{[\s\S]*\}', response.text)
                return json.loads(json_match.group()) if json_match else None
        except Exception:
            return None
    
    @staticmethod
    def analyze_resume_field(text, skills):
        try:
            if not GEMINI_AVAILABLE: return None
            
            prompt = f"""
            Based on the resume text and skills provided, determine the most suitable career field for this candidate.
            Choose from: "Data Science", "Web Development", "Android Development", "IOS Development", "UI-UX Development", or "General"
            
            Also provide 5-8 recommended skills that would complement their profile in this field.
            
            Return the information in JSON format with these keys: 
            - "predicted_field": the career field name as a string
            - "recommended_skills": array of strings with recommended skills
            
            Resume text excerpt:
            {text[:5000]}
            
            Skills already in resume:
            {', '.join(skills)}
            """
            
            response = model.generate_content(prompt)
            
            try:
                return json.loads(response.text)
            except:
                json_match = re.search(r'\{[\s\S]*\}', response.text)
                return json.loads(json_match.group()) if json_match else None
        except Exception:
            return None
    
    @staticmethod
    def ats_analysis(resume_text, job_description):
        try:
            if not GEMINI_AVAILABLE or not job_description: return None
            
            prompt = f"""
            Analyze how well this resume matches the job description using ATS (Applicant Tracking System) criteria.
            
            Job Description:
            {job_description}
            
            Resume Text (excerpt):
            {resume_text[:7000]}
            
            Return your analysis in JSON format with these keys:
            - "score": a number between 0-100 representing match percentage
            - "key_terms": array of most important keywords from the job description (max 10)
            - "matched_terms": array of keywords from job description found in the resume
            - "missing_terms": array of important keywords missing from the resume
            """
            
            response = model.generate_content(prompt)
            
            try:
                return json.loads(response.text)
            except:
                json_match = re.search(r'\{[\s\S]*\}', response.text)
                return json.loads(json_match.group()) if json_match else None
        except Exception:
            return None

class ATSScorer:
    @staticmethod
    def score_resume(resume_text, job_description):
        gemini_analysis = GeminiProcessor.ats_analysis(resume_text, job_description) if job_description else None
        
        if gemini_analysis:
            return {
                'score': gemini_analysis.get('score', 0),
                'key_terms': gemini_analysis.get('key_terms', []),
                'matched_terms': gemini_analysis.get('matched_terms', []),
                'missing_terms': gemini_analysis.get('missing_terms', [])
            }
        
        if not job_description: return {'score': 0, 'key_terms': [], 'matched_terms': [], 'missing_terms': []}
        
        resume_lower, jd_lower = resume_text.lower(), job_description.lower()
        stop_words = set(stopwords.words('english'))
        jd_words = [word for word in jd_lower.split() if word not in stop_words and len(word) > 2]
        
        matches = sum(1 for word in jd_words if word in resume_lower)
        score = min(100, int((matches / len(jd_words)) * 100)) if jd_words else 0
        key_terms = [word for word in set(jd_words) if len(word) > 5][:10]
        matched_terms = [term for term in key_terms if term in resume_lower]
        
        return {
            'score': score,
            'key_terms': key_terms,
            'matched_terms': matched_terms,
            'missing_terms': [term for term in key_terms if term not in resume_lower]
        }

class ResumeAnalyzer:
    def __init__(self):
        self.field_keywords = {
            'Data Science': ['tensorflow','keras','pytorch','machine learning','deep learning','flask','streamlit', 'data science', 'python', 'statistics'],
            'Web Development': ['react', 'django', 'node jS', 'react js', 'php', 'laravel', 'magento', 'wordpress', 'javascript', 'angular js', 'c#', 'flask', 'web development', 'html', 'css'],
            'Android Development': ['android','android development','flutter','kotlin','xml','kivy', 'mobile development'],
            'IOS Development': ['ios','ios development','swift','cocoa','cocoa touch','xcode', 'objective-c'],
            'UI-UX Development': ['ux','adobe xd','figma','zeplin','balsamiq','ui','prototyping','wireframes','storyframes', 'adobe photoshop','photoshop','editing','adobe illustrator','illustrator','adobe after effects', 'after effects','adobe premier pro','premier pro','adobe indesign','indesign','wireframe', 'solid','grasp','user research','user experience']
        }
        
        self.recommended_skills = {
            'Data Science': ['Data Visualization','Predictive Analysis','Statistical Modeling','Data Mining',
                           'Clustering & Classification','Data Analytics','Quantitative Analysis','Web Scraping',
                           'ML Algorithms','Keras','Pytorch','Probability','Scikit-learn','Tensorflow','Flask','Streamlit'],
            'Web Development': ['React','Django','Node JS','React JS','php','laravel','Magento',
                              'wordpress','Javascript','Angular JS','c#','Flask','SDK'],
            'Android Development': ['Android','Android development','Flutter','Kotlin','XML','Java',
                                  'Kivy','GIT','SDK','SQLite'],
            'IOS Development': ['IOS','IOS Development','Swift','Cocoa','Cocoa Touch','Xcode',
                              'Objective-C','SQLite','Plist','StoreKit',"UI-Kit",'AV Foundation','Auto-Layout'],
            'UI-UX Development': ['UI','User Experience','Adobe XD','Figma','Zeplin','Balsamiq',
                                'Prototyping','Wireframes','Storyframes','Adobe Photoshop',
                                'Editing','Illustrator','After Effects','Premier Pro',
                                'Indesign','Wireframe','Solid','Grasp','User Research']
        }
        
        self.course_mapping = {
            'Data Science': ds_course, 'Web Development': web_course,
            'Android Development': android_course, 'IOS Development': ios_course,
            'UI-UX Development': uiux_course
        }
    
    def analyze(self, resume_data):
        gemini_analysis = GeminiProcessor.analyze_resume_field(resume_data['text'], resume_data['skills'])
        
        if gemini_analysis:
            return {
                'predicted_field': gemini_analysis.get('predicted_field', 'General'),
                'recommended_skills': gemini_analysis.get('recommended_skills', 
                    ['Communication Skills', 'Problem Solving', 'Critical Thinking', 'Teamwork', 
                     'Time Management', 'Leadership', 'Project Management', 'Adaptability']),
                'skill_match': True if gemini_analysis.get('predicted_field', 'General') != 'General' else False
            }
        
        results = {
            'predicted_field': 'General',
            'recommended_skills': ['Communication Skills', 'Problem Solving', 'Critical Thinking', 'Teamwork', 
                                 'Time Management', 'Leadership', 'Project Management', 'Adaptability'],
            'skill_match': False
        }
        
        for skill in resume_data['skills']:
            for field, keywords in self.field_keywords.items():
                if skill.lower() in keywords:
                    results['predicted_field'] = field
                    results['recommended_skills'] = self.recommended_skills[field]
                    results['skill_match'] = True
                    return results
        return results
    
    def calculate_resume_score(self, resume_text):
        elements = {
            'objective': 'objective' in resume_text.lower() or 'summary' in resume_text.lower(),
            'declaration': 'declaration' in resume_text.lower(),
            'hobbies': 'hobbies' in resume_text.lower() or 'interests' in resume_text.lower(),
            'achievements': 'achievements' in resume_text.lower() or 'achievement' in resume_text.lower(),
            'projects': 'projects' in resume_text.lower() or 'project' in resume_text.lower()
        }
        return sum(20 for present in elements.values() if present)

class Utils:
    @staticmethod
    def show_pdf(file_path):
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        st.markdown(f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>', unsafe_allow_html=True)

    @staticmethod
    def get_table_download_link(df, filename, text):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        return f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'

    @staticmethod
    def course_recommender(course_list):
        st.subheader("Courses & Certificates Recommendations 🎓")
        rec_course = []
        no_of_reco = st.slider('Choose Number of Course Recommendations:', 1, 10, 5)
        random.shuffle(course_list)
        
        for i, (c_name, c_link) in enumerate(course_list[:no_of_reco], 1):
            st.markdown(f"({i}) [{c_name}]({c_link})")
            rec_course.append(c_name)
        return rec_course

def user_section(db):
    st.markdown('Upload your resume, and get smart recommendations')
    
    # File uploader
    pdf_file = st.file_uploader("Choose your Resume", type=["pdf"])
    
    # ATS checkbox
    use_ats = st.checkbox("Use ATS Scoring", value=False)
    
    job_description = ""
    if use_ats:
        with st.expander("🔍 ATS Resume Scoring"):
            job_description = st.text_area("Enter the job description to match against your resume:", height=150)
    
    if pdf_file is not None:
        with st.spinner('Analyzing your Resume...'):
            time.sleep(1)
        
        save_path = './Uploaded_Resumes/' + pdf_file.name
        with open(save_path, "wb") as f:
            f.write(pdf_file.getbuffer())
        
        with st.expander("📄 View Uploaded Resume"):
            Utils.show_pdf(save_path)
            
        resume_data = ResumeParser.parse_resume(save_path)
        
        if resume_data:
            resume_text = resume_data['text']
            analyzer = ResumeAnalyzer()
            ats_results = ATSScorer.score_resume(resume_text, job_description) if use_ats and job_description else None
            
            st.markdown("---")
            st.header("Resume Analysis")
            st.success(f"Hello {resume_data['name']}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Your Basic info")
                st.write(f"📝 **Name:** {resume_data['name']}")
                st.write(f"📧 **Email:** {resume_data['email']}")
                st.write(f"📱 **Contact:** {resume_data['mobile_number']}")
                st.write(f"📄 **Resume pages:** {resume_data['no_of_pages']}")
            
            with col2:
                cand_level = "Fresher" if resume_data['no_of_pages'] == 1 else "Intermediate" if resume_data['no_of_pages'] == 2 else "Experienced"
                st.info(f"You are at **{cand_level}** level")
            
            # Display current skills
            st.subheader("🔧 Your Current Skills")
            if resume_data['skills']:
                st.write(", ".join(resume_data['skills']))
            else:
                st.info("No skills detected")
            
            analysis_results = analyzer.analyze(resume_data)
            reco_field = analysis_results['predicted_field']
            
            if analysis_results['skill_match']:
                st.success(f"Our analysis shows you're looking for **{reco_field}** Jobs.")
            else:
                st.warning("We couldn't determine your specific field. Here are some general recommendations.")
            
            # Display recommended skills
            st.subheader("✨ Recommended Skills")
            if analysis_results['recommended_skills']:
                st.write(", ".join(analysis_results['recommended_skills']))
                st.info("Adding these skills to your resume will boost your chances of getting hired")
            else:
                st.info("No specific skill recommendations available")
            
            courses = analyzer.course_mapping.get(reco_field, ds_course)
            rec_course = Utils.course_recommender(courses)
            
            st.markdown("---")
            st.subheader("📝 Resume Improvement Tips")
            
            col1, col2 = st.columns(2)
            
            with col1:
                elements = {
                    'Objective/Summary': 'objective' in resume_text.lower() or 'summary' in resume_text.lower(),
                    'Declaration': 'declaration' in resume_text.lower(),
                    'Hobbies/Interests': 'hobbies' in resume_text.lower() or 'interests' in resume_text.lower(),
                    'Achievements': 'achievements' in resume_text.lower() or 'achievement' in resume_text.lower(),
                    'Projects': 'projects' in resume_text.lower() or 'project' in resume_text.lower()
                }
                for element, present in elements.items():
                    st.write(f"{'✅' if present else '❌'} **{element}**")
            
            with col2:
                resume_score = analyzer.calculate_resume_score(resume_text)
                st.subheader("Resume Score")
                st.progress(resume_score/100)
                st.write(f"{resume_score}%")
                st.caption("Based on resume content analysis")
            
            if use_ats and ats_results:
                st.markdown("---")
                st.subheader("📊 ATS Compatibility Analysis")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    ats_score = ats_results['score']
                    message = "Great match! Your resume is highly compatible." if ats_score >= 70 else "Decent match. Consider adding missing keywords." if ats_score >= 40 else "Low compatibility. Review missing keywords carefully."
                    
                    st.write(f"**ATS Score: {ats_score}%**")
                    st.write(message)
                    st.progress(ats_score/100)
                
                with col2:
                    tab1, tab2 = st.tabs(["Matched Keywords ✅", "Missing Keywords ❌"])
                    with tab1:
                        st.write("\n".join(f"- {term}" for term in ats_results['matched_terms']) if ats_results['matched_terms'] else "No significant keywords matched")
                    with tab2:
                        st.write("\n".join(f"- {term}" for term in ats_results['missing_terms']) if ats_results['missing_terms'] else "Your resume contains all significant keywords!")
            
            st.balloons()
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
            data_tuple = (
                resume_data['name'], resume_data['email'], str(resume_score), timestamp,
                str(resume_data['no_of_pages']), reco_field, cand_level, str(resume_data['skills']),
                str(analysis_results['recommended_skills']), str(rec_course)
            )
            if use_ats and ats_results:
                data_tuple += (str(ats_results['score']),)
            db.insert_data(data_tuple)
            
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📝 Resume Writing Tips")
                st.video(random.choice(resume_videos))
            with col2:
                st.subheader("🎯 Interview Tips")
                st.video(random.choice(interview_videos))
        else:
            st.error('Something went wrong while parsing the resume.')

def admin_section(db):
    st.header("AI Resume Analyzer Admin Dashboard")
    
    if 'admin_logged_in' not in st.session_state:
        st.session_state.admin_logged_in = False
    
    if not st.session_state.admin_logged_in:
        with st.form("login_form"):
            st.subheader("🔒 Admin Login")
            ad_user = st.text_input("Username", key="admin_username").strip()
            ad_password = st.text_input("Password", type='password', key="admin_password").strip()
            login_submitted = st.form_submit_button("Login")
            
            if login_submitted:
                if ad_user == 'admin' and ad_password == 'admin1234':
                    st.session_state.admin_logged_in = True
                    st.success("Login successful! Redirecting to admin dashboard...")
                    st.experimental_rerun()
                else:
                    st.error(f"Wrong ID & Password Provided. You entered username: '{ad_user}'")
    
    if st.session_state.admin_logged_in:
        st.success("Welcome Admin! You now have access to the complete dashboard.")
        if st.button("Logout", key="admin_logout_button"):
            st.session_state.admin_logged_in = False
            st.experimental_rerun()
        
        tab1, tab2, tab3 = st.tabs(["📊 User Data", "📑 Batch Resume Analysis", "📈 Analytics"])
        
        with tab1:
            st.header("User's Data")
            df = db.fetch_all_data()
            
            if not df.empty:
                required_columns = ['name', 'email', 'predicted_field', 'user_level', 'actual_skills', 'resume_score']
                if 'ats_score' in df.columns: required_columns.append('ats_score')
                
                df_display = df[required_columns].copy()
                column_mapping = {
                    'name': 'Name', 'email': 'Email', 'predicted_field': 'Predicted Field',
                    'user_level': 'User Level', 'actual_skills': 'Current Skills',
                    'resume_score': 'Resume Score'
                }
                if 'ats_score' in df.columns: column_mapping['ats_score'] = 'ATS Score'
                
                df_display = df_display.rename(columns=column_mapping)
                st.dataframe(df_display, use_container_width=True)
                st.markdown(Utils.get_table_download_link(df_display, 'User_Data.csv', 'Download Report'), unsafe_allow_html=True)
            else:
                st.info("No data available")
        
        with tab2:
            st.header("📑 Batch Resume Analysis")
            job_desc = st.text_area("Enter the job description:", height=150, key="batch_job_desc")
            uploaded_files = st.file_uploader("Upload resumes (PDF format only)", type=["pdf"], 
                                            accept_multiple_files=True, key="batch_resume_uploader")
            if uploaded_files:
                st.success(f"{len(uploaded_files)} resumes uploaded successfully")
            
            if st.button("Analyze Resumes", key="analyze_resumes_button") and uploaded_files and job_desc:
                progress_bar = st.progress(0)
                results_container = st.container()
                ats_results = []
                
                for i, file in enumerate(uploaded_files):
                    try:
                        progress_bar.progress((i + 1) / len(uploaded_files))
                        save_path = f'./Uploaded_Resumes/{file.name}'
                        with open(save_path, "wb") as f:
                            f.write(file.getbuffer())
                        
                        resume_data = ResumeParser.parse_resume(save_path)
                        ats_result = ATSScorer.score_resume(resume_data['text'], job_desc)
                        analysis_results = ResumeAnalyzer().analyze(resume_data)
                        cand_level = 'Fresher' if resume_data['no_of_pages'] == 1 else 'Intermediate' if resume_data['no_of_pages'] == 2 else 'Experienced'
                        
                        ats_results.append({
                            'Name': resume_data['name'], 
                            'Email': resume_data['email'],
                            'ATS Score': ats_result['score'], 
                            'Skills': ", ".join(resume_data['skills']),
                            'File': file.name
                        })
                    except Exception as e:
                        st.error(f"Error processing {file.name}: {str(e)}")
                
                progress_bar.empty()
                if ats_results:
                    with results_container:
                        st.success("Analysis Complete!")
                        results_df = pd.DataFrame(ats_results).sort_values(by='ATS Score', ascending=False)
                        st.subheader("🏆 Top Candidates")
                        st.dataframe(results_df, use_container_width=True)
                        st.markdown(Utils.get_table_download_link(results_df, 'resume_analysis_results.csv', 'Download Analysis Results'), unsafe_allow_html=True)
                        
                        st.subheader("🔍 Detailed View of Top Candidates")
                        num_top = min(5, len(results_df))
                        top_candidates = results_df.head(num_top)
                        if num_top > 0:
                            top_tabs = st.tabs([f"#{i+1}: {row['Name']}" for i, (_, row) in enumerate(top_candidates.iterrows())])
                            
                            for i, (tab, (_, candidate)) in enumerate(zip(top_tabs, top_candidates.iterrows())):
                                with tab:
                                    file_path = f"./Uploaded_Resumes/{candidate['File']}"
                                    resume_data = ResumeParser.parse_resume(file_path)
                                    ats_result = ATSScorer.score_resume(resume_data['text'], job_desc)
                                    analysis_results = ResumeAnalyzer().analyze(resume_data)
                                    cand_level = 'Fresher' if resume_data['no_of_pages'] == 1 else 'Intermediate' if resume_data['no_of_pages'] == 2 else 'Experienced'
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.subheader(f"Candidate: {candidate['Name']}")
                                        st.write(f"📧 **Email:** {candidate['Email']}")
                                        st.write(f"🎯 **Experience Level:** {cand_level}")
                                        st.write(f"📊 **Field:** {analysis_results['predicted_field']}")
                                        st.write(f"🏅 **ATS Score:** {ats_result['score']}%")
                                        if st.button(f"View Resume", key=f"view_resume_{i}"):
                                            Utils.show_pdf(file_path)
                                    
                                    with col2:
                                        skill_tab, matched_tab, missing_tab = st.tabs(["Skills", "Matched Keywords", "Missing Keywords"])
                                        with skill_tab:
                                            skills_list = resume_data['skills']
                                            st.markdown("\n".join(f"- {item}" for item in skills_list) if skills_list else "No skills detected")
                                        with matched_tab:
                                            matched_keywords = ats_result['matched_terms']
                                            st.markdown("\n".join(f"- {item}" for item in matched_keywords) if matched_keywords else "No matched keywords found")
                                        with missing_tab:
                                            missing_keywords = ats_result['missing_terms']
                                            st.markdown("\n".join(f"- {item}" for item in missing_keywords) if missing_keywords else "No missing keywords")
                else:
                    st.error("❌ No results were generated. Please check the files and try again.")
        
        with tab3:
            st.header("📈 Data Visualizations")
            df = db.fetch_all_data()
            if not df.empty:
                viz_col1, viz_col2 = st.columns(2)
                
                with viz_col1:
                    field_counts = df['predicted_field'].value_counts()
                    st.subheader("Field Distribution")
                    if len(field_counts) > 0:
                        fig = px.pie(values=field_counts.values, names=field_counts.index, title='Predicted Fields', 
                                    hole=0.4, color_discrete_sequence=px.colors.sequential.Purp_r)
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("ℹ️ Not enough data for visualization")
                
                with viz_col2:
                    level_counts = df['user_level'].value_counts()
                    st.subheader("Experience Level")
                    if len(level_counts) > 0:
                        fig = px.pie(values=level_counts.values, names=level_counts.index, title="Experience Distribution", 
                                    hole=0.4, color_discrete_sequence=px.colors.sequential.Viridis_r)
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("ℹ️ Not enough data for visualization")
                        

def main():
    st.set_page_config(
        page_title="AI Resume Analyzer", 
        page_icon='./Logo/logo2.png', 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Create database instance
    db = Database()
    
    # Create sidebar 
    st.sidebar.image('./Logo/logo2.png', width=300)
    st.sidebar.title("AI Resume Analyzer")
    
    # Create navigation options
    app_mode = st.sidebar.selectbox("Choose Mode", ["User Section", "Admin Section"])
    
    if app_mode == "User Section":
        st.sidebar.info('This AI-powered tool analyzes your resume against job descriptions, recommends skills, and measures ATS compatibility.')
        user_section(db)
    else:
        admin_section(db)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info(
        "This tool uses AI to analyze resumes and help improve job prospects. "
        "Upload your resume (PDF format only) to get personalized recommendations."
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This application is designed to help job seekers optimize their resumes "
        "and get personalized recommendations to improve their chances in the job market."
    )
    
    # Create necessary directory if it doesn't exist
    os.makedirs('./Uploaded_Resumes', exist_ok=True)

# Call the main function
if __name__ == '__main__':
    main()
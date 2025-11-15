import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Military Personnel Analysis",
    page_icon="🎖️",
    layout="wide"
)

# Title and description
st.title("🎖️ Military Personnel Performance Analysis Dashboard")
st.markdown("---")

# Sidebar for file upload
st.sidebar.header("📁 Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload Military Personnel CSV", type=['csv'])

if uploaded_file is not None:
    try:
        # Load data
        df = pd.read_csv(uploaded_file)
        
        st.success("✅ Data loaded successfully!")
        
        # Display data overview
        st.header("📊 Data Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Personnel", len(df))
        with col2:
            st.metric("Total Units", df['Unit'].nunique())
        with col3:
            st.metric("Features", len(df.columns))
        with col4:
            st.metric("Missing Values", df.isnull().sum().sum())
        
        # Show raw data
        with st.expander("📋 View Raw Data"):
            st.dataframe(df.head(20), use_container_width=True)
        
        # Data Info
        with st.expander("ℹ️ Data Information"):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Data Types")
                st.dataframe(df.dtypes.to_frame('Data Type'), use_container_width=True)
            with col2:
                st.subheader("Missing Values")
                missing = df.isnull().sum().to_frame('Missing Count')
                missing['Percentage'] = (missing['Missing Count'] / len(df) * 100).round(2)
                st.dataframe(missing, use_container_width=True)
        
        with st.expander("📈 Statistical Summary"):
            st.dataframe(df.describe(), use_container_width=True)
        
        # =======================
        # 🔧 Data Cleaning
        # =======================
        st.header("🔧 Data Cleaning & Preprocessing")
        
        with st.spinner("Cleaning data..."):
            # Store original counts
            original_missing = df.isnull().sum().sum()
            
            # Fix typos in Unit column
            df['Unit'] = df['Unit'].str.strip().str.capitalize()
            df['Unit'] = df['Unit'].replace({'Alphaa': 'Alpha'})
            
            # Fix typos in Training_Type
            df['Training_Type'] = df['Training_Type'].replace({'Tehc': 'Tech'})
            
            # Fill missing numeric values with column median
            df['Training_Score'] = df['Training_Score'].fillna(df['Training_Score'].median())
            df['Mission_Success_Rate'] = df['Mission_Success_Rate'].fillna(df['Mission_Success_Rate'].median())
            
            # Calculate cleaning results
            final_missing = df.isnull().sum().sum()
            fixed_missing = original_missing - final_missing
        
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"✅ Fixed typos in Unit and Training_Type columns")
        with col2:
            st.success(f"✅ Filled {fixed_missing} missing values with median")
        
        st.markdown("---")
        
        # =======================
        # 📊 Visualizations
        # =======================
        st.header("📊 Performance Visualizations")
        
        # Calculate average training score per unit
        avg_scores = df.groupby('Unit')['Training_Score'].mean()
        
        # Row 1: Pie Chart
        st.subheader("🥧 Training Score Distribution by Unit")
        fig1, ax1 = plt.subplots(figsize=(8, 8))
        colors = sns.color_palette('Set3', len(avg_scores))
        ax1.pie(avg_scores, labels=avg_scores.index, autopct='%1.1f%%', 
                startangle=140, colors=colors)
        ax1.set_title('Proportion of Average Training Scores by Unit', fontsize=14, fontweight='bold')
        ax1.axis('equal')
        st.pyplot(fig1)
        
        st.markdown("---")
        
        # Row 2: Two columns for bar plots
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Average Fatigue Level by Unit")
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            avg_fatigue = df.groupby('Unit')['Fatigue_Level'].mean().reset_index()
            sns.barplot(x='Unit', y='Fatigue_Level', data=avg_fatigue, palette='magma', ax=ax2)
            ax2.set_title('Average Fatigue Level by Unit', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Avg Fatigue Level (1 = Fresh, 10 = Very Fatigued)')
            ax2.set_xlabel('Unit')
            plt.tight_layout()
            st.pyplot(fig2)
        
        with col2:
            st.subheader("📈 Average Mission Success Rate by Unit")
            fig3, ax3 = plt.subplots(figsize=(8, 5))
            avg_success = df.groupby('Unit')['Mission_Success_Rate'].mean().reset_index()
            sns.lineplot(x='Unit', y='Mission_Success_Rate', data=avg_success, 
                        marker='o', linewidth=2.5, markersize=10, ax=ax3)
            ax3.set_title('Average Mission Success Rate by Unit', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Avg Mission Success Rate (%)')
            ax3.set_xlabel('Unit')
            plt.tight_layout()
            st.pyplot(fig3)
        
        st.markdown("---")
        
        # =======================
        # 📋 Performance Analysis
        # =======================
        st.header("📋 Performance Analysis")
        
        # Group by Unit to find underperformers
        unit_summary = df.groupby('Unit').agg({
            'Training_Score': 'mean',
            'Mission_Success_Rate': 'mean',
            'Fatigue_Level': 'mean',
            'Attendance_Rate': 'mean'
        }).reset_index().sort_values(by='Training_Score')
        
        # Round values for better display
        unit_summary['Training_Score'] = unit_summary['Training_Score'].round(2)
        unit_summary['Mission_Success_Rate'] = unit_summary['Mission_Success_Rate'].round(2)
        unit_summary['Fatigue_Level'] = unit_summary['Fatigue_Level'].round(2)
        unit_summary['Attendance_Rate'] = unit_summary['Attendance_Rate'].round(2)
        
        st.subheader("📊 Performance Summary by Unit")
        
        # Color code the dataframe
        def color_performance(val, col_name):
            if col_name == 'Training_Score':
                if val < 70:
                    return 'background-color: #ffcccc'
                elif val >= 80:
                    return 'background-color: #ccffcc'
            elif col_name == 'Mission_Success_Rate':
                if val < 75:
                    return 'background-color: #ffcccc'
                elif val >= 85:
                    return 'background-color: #ccffcc'
            elif col_name == 'Fatigue_Level':
                if val > 7:
                    return 'background-color: #ffcccc'
                elif val <= 5:
                    return 'background-color: #ccffcc'
            elif col_name == 'Attendance_Rate':
                if val < 85:
                    return 'background-color: #ffcccc'
                elif val >= 95:
                    return 'background-color: #ccffcc'
            return ''
        
        # Apply styling
        styled_summary = unit_summary.style.apply(
            lambda x: [color_performance(v, x.name) for v in x], 
            subset=['Training_Score', 'Mission_Success_Rate', 'Fatigue_Level', 'Attendance_Rate']
        )
        
        st.dataframe(styled_summary, use_container_width=True)
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Training Score", f"{df['Training_Score'].mean():.2f}")
        with col2:
            st.metric("Avg Mission Success", f"{df['Mission_Success_Rate'].mean():.2f}%")
        with col3:
            st.metric("Avg Fatigue Level", f"{df['Fatigue_Level'].mean():.2f}")
        with col4:
            st.metric("Avg Attendance", f"{df['Attendance_Rate'].mean():.2f}%")
        
        st.markdown("---")
        
        # Flag units with below-average training or mission scores
        avg_training = df['Training_Score'].mean()
        avg_mission = df['Mission_Success_Rate'].mean()
        
        underperforming_units = unit_summary[
            (unit_summary['Training_Score'] < avg_training) |
            (unit_summary['Mission_Success_Rate'] < avg_mission)
        ]
        
        st.subheader("🚨 Units Requiring Additional Training/Support")
        
        if len(underperforming_units) > 0:
            st.warning(f"Found {len(underperforming_units)} unit(s) that need attention:")
            
            for idx, row in underperforming_units.iterrows():
                with st.expander(f"⚠️ Unit: {row['Unit']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Training Score", f"{row['Training_Score']:.2f}", 
                                 delta=f"{row['Training_Score'] - avg_training:.2f} vs avg",
                                 delta_color="inverse")
                        st.metric("Fatigue Level", f"{row['Fatigue_Level']:.2f}")
                    with col2:
                        st.metric("Mission Success Rate", f"{row['Mission_Success_Rate']:.2f}%",
                                 delta=f"{row['Mission_Success_Rate'] - avg_mission:.2f}% vs avg",
                                 delta_color="inverse")
                        st.metric("Attendance Rate", f"{row['Attendance_Rate']:.2f}%")
                    
                    # Recommendations
                    st.markdown("**Recommendations:**")
                    recommendations = []
                    if row['Training_Score'] < avg_training:
                        recommendations.append("- 📚 Increase training frequency and quality")
                    if row['Mission_Success_Rate'] < avg_mission:
                        recommendations.append("- 🎯 Review mission preparation protocols")
                    if row['Fatigue_Level'] > 7:
                        recommendations.append("- 😴 Implement better rest schedules")
                    if row['Attendance_Rate'] < 85:
                        recommendations.append("- 📋 Address attendance issues")
                    
                    for rec in recommendations:
                        st.markdown(rec)
        else:
            st.success("✅ All units are performing at or above average!")
        
        st.markdown("---")
        
        # Additional Insights
        st.header("💡 Additional Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Top Performing Unit")
            top_unit = unit_summary.loc[unit_summary['Training_Score'].idxmax()]
            st.success(f"**{top_unit['Unit']}**")
            st.write(f"- Training Score: {top_unit['Training_Score']:.2f}")
            st.write(f"- Mission Success: {top_unit['Mission_Success_Rate']:.2f}%")
            st.write(f"- Fatigue Level: {top_unit['Fatigue_Level']:.2f}")
        
        with col2:
            st.subheader("⚠️ Unit Needing Most Support")
            bottom_unit = unit_summary.loc[unit_summary['Training_Score'].idxmin()]
            st.error(f"**{bottom_unit['Unit']}**")
            st.write(f"- Training Score: {bottom_unit['Training_Score']:.2f}")
            st.write(f"- Mission Success: {bottom_unit['Mission_Success_Rate']:.2f}%")
            st.write(f"- Fatigue Level: {bottom_unit['Fatigue_Level']:.2f}")
        
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("Please make sure your CSV file has the correct format with columns: Unit, Training_Score, Mission_Success_Rate, Fatigue_Level, Attendance_Rate, Training_Type")

else:
    # Instructions when no file is uploaded
    st.info("👈 Please upload a CSV file to begin analysis")
    
    st.markdown("""
    ### 📋 Required CSV Format:
    Your CSV file should contain the following columns:
    - **Personnel_ID**: Unique identifier for each person
    - **Unit**: Military unit (e.g., Alpha, Bravo, Charlie)
    - **Training_Score**: Training performance score (0-100)
    - **Mission_Success_Rate**: Mission success percentage (0-100)
    - **Fatigue_Level**: Fatigue level (1-10, where 1=Fresh, 10=Very Fatigued)
    - **Attendance_Rate**: Attendance percentage (0-100)
    - **Training_Type**: Type of training received
    
    ### 🚀 Features:
    - ✅ Automated data cleaning and typo correction
    - ✅ Interactive performance visualizations
    - ✅ Unit-wise performance analysis
    - ✅ Identification of underperforming units
    - ✅ Actionable recommendations
    - ✅ Color-coded performance metrics
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Military Personnel Analysis Dashboard")
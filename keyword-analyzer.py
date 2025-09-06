import streamlit as st
import pandas as pd
import re
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import base64
import zipfile
import tempfile
import os

def extract_keywords(text, keywords, case_sensitive=False, whole_words=True):
    """Extract keyword counts and positions from text."""
    if not text.strip() or not keywords:
        return []
    
    # Process keywords
    keyword_list = [k.strip() for k in keywords.split(',') if k.strip()]
    search_text = text if case_sensitive else text.lower()
    
    results = []
    
    for keyword in keyword_list:
        search_term = keyword if case_sensitive else keyword.lower()
        matches = []
        count = 0
        
        if whole_words:
            # Use regex for whole word matching
            pattern = r'\b' + re.escape(search_term) + r'\b'
            flags = 0 if case_sensitive else re.IGNORECASE
            
            for match in re.finditer(pattern, text, flags):
                matches.append({
                    'start': match.start(),
                    'end': match.end(),
                    'text': match.group()
                })
                count += 1
        else:
            # Simple substring search
            start = 0
            while True:
                pos = search_text.find(search_term, start)
                if pos == -1:
                    break
                matches.append({
                    'start': pos,
                    'end': pos + len(search_term),
                    'text': text[pos:pos + len(search_term)]
                })
                count += 1
                start = pos + 1
        
        # Calculate percentage and density
        percentage = (count * len(search_term) / len(text)) * 100 if text else 0
        density_per_1000 = (count / len(text.split())) * 1000 if text.split() else 0
        
        results.append({
            'keyword': keyword,
            'count': count,
            'matches': matches,
            'percentage': round(percentage, 2),
            'density': round(density_per_1000, 2)
        })
    
    # Sort by count (descending)
    return sorted(results, key=lambda x: x['count'], reverse=True)

def process_multiple_transcripts(transcripts_dict, keywords, case_sensitive=False, whole_words=True):
    """Process multiple transcripts and return combined results."""
    all_results = {}
    
    for name, text in transcripts_dict.items():
        results = extract_keywords(text, keywords, case_sensitive, whole_words)
        all_results[name] = {
            'results': results,
            'word_count': len(text.split()) if text else 0,
            'char_count': len(text) if text else 0,
            'total_keywords': sum(r['count'] for r in results)
        }
    
    return all_results

def create_comparison_dataframe(all_results, keywords):
    """Create a dataframe for comparison across transcripts."""
    keyword_list = [k.strip() for k in keywords.split(',') if k.strip()]
    
    comparison_data = []
    
    for transcript_name, data in all_results.items():
        row = {'Transcript': transcript_name}
        
        # Add basic stats
        row['Word Count'] = data['word_count']
        row['Total Keywords'] = data['total_keywords']
        row['Keyword Density'] = round((data['total_keywords'] / data['word_count']) * 1000, 2) if data['word_count'] > 0 else 0
        
        # Add individual keyword counts
        results_dict = {r['keyword']: r['count'] for r in data['results']}
        for keyword in keyword_list:
            row[f"{keyword}_count"] = results_dict.get(keyword, 0)
            row[f"{keyword}_density"] = round((results_dict.get(keyword, 0) / data['word_count']) * 1000, 2) if data['word_count'] > 0 else 0
        
        comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)

def create_comparison_charts(comparison_df, keywords):
    """Create comparison visualizations."""
    charts = {}
    
    if comparison_df.empty:
        return charts
    
    keyword_list = [k.strip() for k in keywords.split(',') if k.strip()]
    
    # 1. Grouped bar chart for keyword counts
    count_cols = [f"{k}_count" for k in keyword_list]
    if all(col in comparison_df.columns for col in count_cols):
        fig_counts = go.Figure()
        
        for keyword in keyword_list:
            fig_counts.add_trace(go.Bar(
                name=keyword,
                x=comparison_df['Transcript'],
                y=comparison_df[f"{keyword}_count"],
                text=comparison_df[f"{keyword}_count"],
                textposition='auto',
            ))
        
        fig_counts.update_layout(
            title='Keyword Count Comparison Across Transcripts',
            xaxis_title='Transcripts',
            yaxis_title='Count',
            barmode='group',
            height=500
        )
        charts['counts'] = fig_counts
    
    # 2. Heatmap for keyword densities
    density_cols = [f"{k}_density" for k in keyword_list]
    if all(col in comparison_df.columns for col in density_cols):
        density_data = comparison_df[['Transcript'] + density_cols].set_index('Transcript')
        density_data.columns = keyword_list
        
        fig_heatmap = px.imshow(
            density_data.T,
            labels=dict(x="Transcripts", y="Keywords", color="Density (per 1000 words)"),
            title="Keyword Density Heatmap",
            aspect="auto",
            color_continuous_scale='viridis'
        )
        fig_heatmap.update_layout(height=400)
        charts['heatmap'] = fig_heatmap
    
    # 3. Total keyword density comparison
    if 'Keyword Density' in comparison_df.columns:
        fig_density = px.bar(
            comparison_df,
            x='Transcript',
            y='Keyword Density',
            title='Overall Keyword Density Comparison',
            labels={'Keyword Density': 'Keywords per 1000 words'},
            text='Keyword Density'
        )
        fig_density.update_traces(textposition='outside')
        fig_density.update_layout(height=400)
        charts['overall_density'] = fig_density
    
    # 4. Word count comparison
    if 'Word Count' in comparison_df.columns:
        fig_words = px.bar(
            comparison_df,
            x='Transcript',
            y='Word Count',
            title='Transcript Length Comparison',
            text='Word Count'
        )
        fig_words.update_traces(textposition='outside')
        fig_words.update_layout(height=400)
        charts['word_count'] = fig_words
    
    return charts

def highlight_text(text, results):
    """Add HTML highlighting to text based on keyword matches."""
    if not text or not results:
        return text
    
    # Collect all matches with colors
    all_matches = []
    colors = ['#FFE4E1', '#E1F5FE', '#E8F5E8', '#FFF3E0', '#F3E5F5', 
              '#E0F2F1', '#FFF8E1', '#F1F8E9', '#FCE4EC', '#E3F2FD']
    
    for i, result in enumerate(results):
        color = colors[i % len(colors)]
        for match in result['matches']:
            all_matches.append({
                'start': match['start'],
                'end': match['end'],
                'text': match['text'],
                'color': color,
                'keyword': result['keyword']
            })
    
    # Sort by position (descending) to avoid position shifts
    all_matches.sort(key=lambda x: x['start'], reverse=True)
    
    # Apply highlighting
    highlighted = text
    for match in all_matches:
        before = highlighted[:match['start']]
        after = highlighted[match['end']:]
        highlighted_text = f'<mark style="background-color: {match["color"]}; padding: 2px 4px; border-radius: 3px; font-weight: 500;" title="{match["keyword"]}">{match["text"]}</mark>'
        highlighted = before + highlighted_text + after
    
    return highlighted

def export_comparison_to_csv(comparison_df):
    """Export comparison results to CSV."""
    return comparison_df.to_csv(index=False)

def load_transcripts_from_files(uploaded_files):
    """Load transcripts from multiple uploaded files."""
    transcripts = {}
    
    for uploaded_file in uploaded_files:
        try:
            file_name = uploaded_file.name
            
            if uploaded_file.type == "text/plain":
                content = str(uploaded_file.read(), "utf-8")
                transcripts[file_name] = content
            elif uploaded_file.type == "text/csv" or uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                text_columns = df.select_dtypes(include=['object']).columns
                if len(text_columns) > 0:
                    # Use first text column or let user select
                    content = ' '.join(df[text_columns[0]].astype(str))
                    transcripts[file_name] = content
                else:
                    st.warning(f"No text columns found in {file_name}")
        except Exception as e:
            st.error(f"Error loading {uploaded_file.name}: {str(e)}")
    
    return transcripts

# Streamlit App
def main():
    st.set_page_config(
        page_title="Multi-Transcript Keyword Analysis",
        page_icon="📊",
        layout="wide"
    )
    
    # Header
    st.title("📊 Multi-Transcript Keyword Analysis Tool")
    st.markdown("Analyze and compare keyword frequencies across multiple transcripts")
    
    # Sidebar for settings
    st.sidebar.header("⚙️ Settings")
    case_sensitive = st.sidebar.checkbox("Case sensitive", value=False)
    whole_words = st.sidebar.checkbox("Whole words only", value=True)
    
    # Analysis mode selection
    analysis_mode = st.sidebar.radio(
        "Analysis Mode",
        ["Single Transcript", "Multiple Transcripts"],
        help="Choose between single transcript analysis or batch comparison"
    )
    
    # Initialize session state
    if 'transcripts' not in st.session_state:
        st.session_state.transcripts = {}
    if 'keywords' not in st.session_state:
        st.session_state.keywords = ""
    
    # Input section
    st.header("📝 Input Data")
    
    if analysis_mode == "Single Transcript":
        # Single transcript mode (original functionality)
        col1, col2 = st.columns([1, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Upload transcript file (optional)", 
                type=['txt', 'csv'],
                help="Upload a text file or CSV with transcript content"
            )
            
            if uploaded_file:
                transcripts = load_transcripts_from_files([uploaded_file])
                if transcripts:
                    st.session_state.transcripts = transcripts
            
            # Text input
            transcript_text = st.text_area(
                "Or paste transcript text",
                height=200,
                placeholder="Paste your transcript here...",
                value=list(st.session_state.transcripts.values())[0] if st.session_state.transcripts else ""
            )
            
            if transcript_text:
                st.session_state.transcripts = {"Manual Input": transcript_text}
        
        with col2:
            keywords = st.text_input(
                "Keywords (comma-separated)",
                placeholder="keyword1, keyword2, keyword3",
                value=st.session_state.keywords
            )
            st.session_state.keywords = keywords
    
    else:
        # Multiple transcripts mode
        st.subheader("📁 Upload Multiple Transcripts")
        
        uploaded_files = st.file_uploader(
            "Upload transcript files",
            type=['txt', 'csv'],
            accept_multiple_files=True,
            help="Upload multiple text files or CSVs for comparison"
        )
        
        if uploaded_files:
            transcripts = load_transcripts_from_files(uploaded_files)
            st.session_state.transcripts = transcripts
            
            if transcripts:
                st.success(f"Loaded {len(transcripts)} transcript(s)")
                with st.expander("Preview loaded transcripts"):
                    for name, content in transcripts.items():
                        st.text(f"{name}: {len(content)} characters, {len(content.split())} words")
        
        # Manual input option for multiple transcripts
        st.subheader("✏️ Or Add Transcripts Manually")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            transcript_name = st.text_input("Transcript Name", placeholder="e.g., Interview 1")
            transcript_text = st.text_area("Transcript Text", height=150, placeholder="Paste transcript content...")
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Add Transcript") and transcript_name and transcript_text:
                st.session_state.transcripts[transcript_name] = transcript_text
                st.success(f"Added '{transcript_name}'")
                st.rerun()
            
            if st.button("Clear All") and st.session_state.transcripts:
                st.session_state.transcripts = {}
                st.success("Cleared all transcripts")
                st.rerun()
        
        # Show current transcripts
        if st.session_state.transcripts:
            st.subheader("📋 Current Transcripts")
            transcript_df = pd.DataFrame([
                {
                    'Name': name,
                    'Characters': len(content),
                    'Words': len(content.split()),
                    'Preview': content[:100] + '...' if len(content) > 100 else content
                }
                for name, content in st.session_state.transcripts.items()
            ])
            st.dataframe(transcript_df, use_container_width=True)
        
        # Keywords input
        keywords = st.text_input(
            "Keywords to analyze (comma-separated)",
            placeholder="keyword1, keyword2, keyword3",
            value=st.session_state.keywords,
            help="Enter keywords you want to search for across all transcripts"
        )
        st.session_state.keywords = keywords
    
    # Analysis section
    if st.session_state.transcripts and st.session_state.keywords:
        st.header("🔍 Analysis Results")
        
        # Process transcripts
        all_results = process_multiple_transcripts(
            st.session_state.transcripts,
            st.session_state.keywords,
            case_sensitive,
            whole_words
        )
        
        if analysis_mode == "Single Transcript":
            # Single transcript analysis (original functionality)
            transcript_name = list(all_results.keys())[0]
            results = all_results[transcript_name]['results']
            
            if results:
                # Summary metrics
                total_keywords = sum(r['count'] for r in results)
                unique_keywords = len([r for r in results if r['count'] > 0])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Occurrences", total_keywords)
                with col2:
                    st.metric("Keywords Found", unique_keywords)
                with col3:
                    st.metric("Keywords Searched", len(results))
                
                # Results table
                df_results = pd.DataFrame([
                    {
                        'Keyword': r['keyword'],
                        'Count': r['count'],
                        'Percentage': f"{r['percentage']}%",
                        'Density (per 1000 words)': r['density']
                    } for r in results
                ])
                
                st.dataframe(df_results, use_container_width=True)
                
                # Visualizations
                st.subheader("📊 Visualizations")
                
                col1, col2 = st.columns(2)
                with col1:
                    fig_bar = px.bar(
                        df_results, 
                        x='Keyword', 
                        y='Count',
                        title='Keyword Frequencies',
                        text='Count'
                    )
                    fig_bar.update_traces(textposition='outside')
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                with col2:
                    if sum(df_results['Count']) > 0:
                        fig_pie = px.pie(
                            df_results[df_results['Count'] > 0],
                            values='Count',
                            names='Keyword',
                            title='Keyword Distribution'
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                
                # Highlighted text
                if st.checkbox("Show Highlighted Transcript", value=True):
                    transcript_text = list(st.session_state.transcripts.values())[0]
                    highlighted = highlight_text(transcript_text, results)
                    
                    st.markdown(
                        f'<div style="max-height: 400px; overflow-y: auto; padding: 10px; background-color: #f8f9fa; border-radius: 5px;">{highlighted}</div>',
                        unsafe_allow_html=True
                    )
        
        else:
            # Multiple transcripts comparison
            comparison_df = create_comparison_dataframe(all_results, st.session_state.keywords)
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Transcripts", len(all_results))
            with col2:
                total_words = sum(data['word_count'] for data in all_results.values())
                st.metric("Total Words", f"{total_words:,}")
            with col3:
                total_keyword_instances = sum(data['total_keywords'] for data in all_results.values())
                st.metric("Total Keyword Instances", total_keyword_instances)
            with col4:
                avg_density = comparison_df['Keyword Density'].mean() if 'Keyword Density' in comparison_df.columns else 0
                st.metric("Avg Keyword Density", f"{avg_density:.2f}")
            
            # Comparison table
            st.subheader("📋 Comparison Table")
            st.dataframe(comparison_df, use_container_width=True)
            
            # Download comparison data
            csv_data = export_comparison_to_csv(comparison_df)
            st.download_button(
                label="📥 Download Comparison CSV",
                data=csv_data,
                file_name="transcript_comparison.csv",
                mime="text/csv"
            )
            
            # Comparison charts
            st.subheader("📊 Comparison Visualizations")
            charts = create_comparison_charts(comparison_df, st.session_state.keywords)
            
            if 'counts' in charts:
                st.plotly_chart(charts['counts'], use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                if 'heatmap' in charts:
                    st.plotly_chart(charts['heatmap'], use_container_width=True)
                if 'overall_density' in charts:
                    st.plotly_chart(charts['overall_density'], use_container_width=True)
            
            with col2:
                if 'word_count' in charts:
                    st.plotly_chart(charts['word_count'], use_container_width=True)
            
            # Individual transcript details
            st.subheader("🔍 Individual Transcript Details")
            selected_transcript = st.selectbox(
                "Select transcript to view details:",
                list(st.session_state.transcripts.keys())
            )
            
            if selected_transcript:
                results = all_results[selected_transcript]['results']
                
                # Individual results table
                df_individual = pd.DataFrame([
                    {
                        'Keyword': r['keyword'],
                        'Count': r['count'],
                        'Percentage': f"{r['percentage']}%",
                        'Density (per 1000 words)': r['density']
                    } for r in results
                ])
                
                st.dataframe(df_individual, use_container_width=True)
                
                # Show highlighted text for selected transcript
                if st.checkbox(f"Show highlighted text for {selected_transcript}"):
                    highlighted = highlight_text(st.session_state.transcripts[selected_transcript], results)
                    st.markdown(
                        f'<div style="max-height: 400px; overflow-y: auto; padding: 10px; background-color: #f8f9fa; border-radius: 5px;">{highlighted}</div>',
                        unsafe_allow_html=True
                    )
    
    elif st.session_state.keywords and not st.session_state.transcripts:
        st.info("Please add some transcripts to analyze.")
    elif st.session_state.transcripts and not st.session_state.keywords:
        st.info("Please enter keywords to search for.")
    else:
        st.info("Please add transcripts and keywords to begin analysis.")
    
    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit • Enhanced for multi-transcript comparison")

if __name__ == "__main__":
    main()

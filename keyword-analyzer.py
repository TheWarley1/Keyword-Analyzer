# Part 1: Core Functions and Enhanced Keyword Extraction
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

def extract_keywords_prediction_market(text, keywords, case_sensitive=False, include_plurals=True, include_compounds=True):
    """
    Extract keyword counts with prediction market rules:
    - Pluralization/possessive forms count
    - Compound words count
    - Other forms do NOT count
    """
    if not text.strip() or not keywords:
        return []
    
    keyword_list = [k.strip() for k in keywords.split(',') if k.strip()]
    results = []
    
    for keyword in keyword_list:
        search_term = keyword if case_sensitive else keyword.lower()
        search_text = text if case_sensitive else text.lower()
        
        matches = []
        patterns = []
        
        # Base pattern for exact word
        base_pattern = r'\b' + re.escape(search_term) + r'\b'
        patterns.append(('exact', base_pattern))
        
        if include_plurals:
            # Plural patterns
            plural_pattern = r'\b' + re.escape(search_term) + r'(?:s|es|ies)\b'
            patterns.append(('plural', plural_pattern))
            
            # Possessive patterns
            possessive_pattern = r'\b' + re.escape(search_term) + r"(?:'s|')\b"
            patterns.append(('possessive', possessive_pattern))
        
        if include_compounds:
            # Compound word patterns (term as part of larger word)
            compound_start = r'\b' + re.escape(search_term) + r'[a-zA-Z]+'
            compound_end = r'[a-zA-Z]+' + re.escape(search_term) + r'\b'
            patterns.append(('compound_start', compound_start))
            patterns.append(('compound_end', compound_end))
        
        all_matches = []
        flags = 0 if case_sensitive else re.IGNORECASE
        
        for pattern_type, pattern in patterns:
            for match in re.finditer(pattern, text, flags):
                # Check for overlaps and avoid duplicates
                overlap = False
                for existing in all_matches:
                    if (match.start() < existing['end'] and match.end() > existing['start']):
                        overlap = True
                        break
                
                if not overlap:
                    all_matches.append({
                        'start': match.start(),
                        'end': match.end(),
                        'text': match.group(),
                        'type': pattern_type,
                        'keyword': keyword
                    })
        
        # Sort by position
        all_matches.sort(key=lambda x: x['start'])
        
        count = len(all_matches)
        percentage = (count * len(search_term) / len(text)) * 100 if text else 0
        density_per_1000 = (count / len(text.split())) * 1000 if text.split() else 0
        
        # Categorize matches by type
        match_types = {
            'exact': [m for m in all_matches if m['type'] == 'exact'],
            'plural': [m for m in all_matches if m['type'] == 'plural'],
            'possessive': [m for m in all_matches if m['type'] == 'possessive'],
            'compound': [m for m in all_matches if m['type'] in ['compound_start', 'compound_end']]
        }
        
        results.append({
            'keyword': keyword,
            'count': count,
            'matches': all_matches,
            'match_types': match_types,
            'percentage': round(percentage, 2),
            'density': round(density_per_1000, 2),
            'market_resolution': 'YES' if count > 0 else 'NO'
        })
    
    return sorted(results, key=lambda x: x['count'], reverse=True)

def validate_compound_word(word, base_term):
    """
    Validate if a word is a true compound containing the base term.
    This helps distinguish real compounds from coincidental letter sequences.
    """
    # Simple validation - could be enhanced with dictionary lookup
    word_lower = word.lower()
    base_lower = base_term.lower()
    
    if base_lower not in word_lower:
        return False
    
    # Check if it's at word boundaries within the compound
    # This is a simplified check - more sophisticated logic could be added
    return len(word) > len(base_term) and word != base_term

def process_multiple_transcripts_market(transcripts_dict, keywords, case_sensitive=False, include_plurals=True, include_compounds=True):
    """Process multiple transcripts with prediction market rules."""
    all_results = {}
    
    for name, text in transcripts_dict.items():
        results = extract_keywords_prediction_market(text, keywords, case_sensitive, include_plurals, include_compounds)
        all_results[name] = {
            'results': results,
            'word_count': len(text.split()) if text else 0,
            'char_count': len(text) if text else 0,
            'total_keywords': sum(r['count'] for r in results),
            'market_resolution': any(r['market_resolution'] == 'YES' for r in results)
        }
    
    return all_results

def create_market_resolution_summary(all_results):
    """Create a summary of market resolution outcomes."""
    summary_data = []
    
    for transcript_name, data in all_results.items():
        resolution = 'YES' if data['market_resolution'] else 'NO'
        
        # Count different match types
        match_type_counts = {'exact': 0, 'plural': 0, 'possessive': 0, 'compound': 0}
        
        for result in data['results']:
            for match_type, matches in result['match_types'].items():
                match_type_counts[match_type] += len(matches)
        
        summary_data.append({
            'Transcript': transcript_name,
            'Market Resolution': resolution,
            'Total Matches': data['total_keywords'],
            'Exact Matches': match_type_counts['exact'],
            'Plural/Possessive': match_type_counts['plural'] + match_type_counts['possessive'],
            'Compound Words': match_type_counts['compound'],
            'Word Count': data['word_count']
        })
    
    return pd.DataFrame(summary_data)

def highlight_text_with_types(text, results):
    """Enhanced highlighting that shows match types with different styles."""
    if not text or not results:
        return text
    
    all_matches = []
    type_colors = {
        'exact': '#FFB6C1',      # Light pink
        'plural': '#98FB98',     # Pale green  
        'possessive': '#87CEEB', # Sky blue
        'compound_start': '#DDA0DD', # Plum
        'compound_end': '#F0E68C'    # Khaki
    }
    
    for result in results:
        for match in result['matches']:
            color = type_colors.get(match['type'], '#FFFFE0')  # Default light yellow
            all_matches.append({
                'start': match['start'],
                'end': match['end'],
                'text': match['text'],
                'color': color,
                'keyword': result['keyword'],
                'match_type': match['type']
            })
    
    # Sort by position (descending) to avoid position shifts
    all_matches.sort(key=lambda x: x['start'], reverse=True)
    
    # Apply highlighting
    highlighted = text
    for match in all_matches:
        before = highlighted[:match['start']]
        after = highlighted[match['end']]
        
        type_label = match['match_type'].replace('_', ' ').title()
        tooltip = f"{match['keyword']} ({type_label})"
        
        highlighted_text = f'<mark style="background-color: {match["color"]}; padding: 2px 4px; border-radius: 3px; font-weight: 500; border: 1px solid #666;" title="{tooltip}">{match["text"]}</mark>'
        highlighted = before + highlighted_text + after
    
    return highlighted

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

# Part 2: Visualization and Comparison Functions

def create_market_comparison_dataframe(all_results, keywords):
    """Create enhanced dataframe for prediction market comparison."""
    keyword_list = [k.strip() for k in keywords.split(',') if k.strip()]
    
    comparison_data = []
    
    for transcript_name, data in all_results.items():
        row = {'Transcript': transcript_name}
        
        # Add basic stats
        row['Word Count'] = data['word_count']
        row['Total Keywords'] = data['total_keywords']
        row['Market Resolution'] = 'YES' if data['market_resolution'] else 'NO'
        row['Keyword Density'] = round((data['total_keywords'] / data['word_count']) * 1000, 2) if data['word_count'] > 0 else 0
        
        # Add individual keyword counts and resolution
        results_dict = {r['keyword']: r for r in data['results']}
        
        for keyword in keyword_list:
            if keyword in results_dict:
                result = results_dict[keyword]
                row[f"{keyword}_total"] = result['count']
                row[f"{keyword}_resolution"] = result['market_resolution']
                row[f"{keyword}_density"] = result['density']
                
                # Breakdown by match type
                row[f"{keyword}_exact"] = len(result['match_types']['exact'])
                row[f"{keyword}_plural"] = len(result['match_types']['plural'])
                row[f"{keyword}_possessive"] = len(result['match_types']['possessive'])
                row[f"{keyword}_compound"] = len(result['match_types']['compound'])
            else:
                row[f"{keyword}_total"] = 0
                row[f"{keyword}_resolution"] = 'NO'
                row[f"{keyword}_density"] = 0
                row[f"{keyword}_exact"] = 0
                row[f"{keyword}_plural"] = 0
                row[f"{keyword}_possessive"] = 0
                row[f"{keyword}_compound"] = 0
        
        comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)

def create_market_resolution_chart(comparison_df, keywords):
    """Create market resolution visualization."""
    keyword_list = [k.strip() for k in keywords.split(',') if k.strip()]
    
    # Prepare data for resolution chart
    resolution_data = []
    
    for _, row in comparison_df.iterrows():
        for keyword in keyword_list:
            resolution_col = f"{keyword}_resolution"
            if resolution_col in comparison_df.columns:
                resolution_data.append({
                    'Transcript': row['Transcript'],
                    'Keyword': keyword,
                    'Resolution': row[resolution_col],
                    'Count': row[f"{keyword}_total"] if f"{keyword}_total" in comparison_df.columns else 0
                })
    
    if not resolution_data:
        return None
    
    resolution_df = pd.DataFrame(resolution_data)
    
    # Create heatmap showing YES/NO resolutions
    pivot_df = resolution_df.pivot(index='Keyword', columns='Transcript', values='Resolution')
    
    # Convert YES/NO to 1/0 for heatmap
    numeric_df = pivot_df.replace({'YES': 1, 'NO': 0})
    
    fig = px.imshow(
        numeric_df,
        labels=dict(x="Transcripts", y="Keywords", color="Market Resolution"),
        title="Market Resolution Heatmap (YES=1, NO=0)",
        aspect="auto",
        color_continuous_scale=[[0, '#ff4444'], [1, '#44ff44']],  # Red for NO, Green for YES
        text_auto=True
    )
    
    # Add custom text annotations
    for i, keyword in enumerate(numeric_df.index):
        for j, transcript in enumerate(numeric_df.columns):
            value = numeric_df.iloc[i, j]
            text = 'YES' if value == 1 else 'NO'
            fig.add_annotation(
                x=j, y=i,
                text=text,
                showarrow=False,
                font=dict(color='white', size=12, family='Arial Black')
            )
    
    fig.update_layout(height=400)
    return fig

def create_match_type_breakdown_chart(comparison_df, keywords):
    """Create stacked bar chart showing match type breakdown."""
    keyword_list = [k.strip() for k in keywords.split(',') if k.strip()]
    
    chart_data = []
    
    for _, row in comparison_df.iterrows():
        for keyword in keyword_list:
            exact_col = f"{keyword}_exact"
            plural_col = f"{keyword}_plural"
            possessive_col = f"{keyword}_possessive"
            compound_col = f"{keyword}_compound"
            
            if all(col in comparison_df.columns for col in [exact_col, plural_col, possessive_col, compound_col]):
                chart_data.append({
                    'Transcript': row['Transcript'],
                    'Keyword': keyword,
                    'Exact': row[exact_col],
                    'Plural': row[plural_col],
                    'Possessive': row[possessive_col],
                    'Compound': row[compound_col]
                })
    
    if not chart_data:
        return None
    
    chart_df = pd.DataFrame(chart_data)
    
    # Create stacked bar chart
    fig = go.Figure()
    
    colors = {
        'Exact': '#FFB6C1',
        'Plural': '#98FB98', 
        'Possessive': '#87CEEB',
        'Compound': '#DDA0DD'
    }
    
    for match_type in ['Exact', 'Plural', 'Possessive', 'Compound']:
        for keyword in keyword_list:
            keyword_data = chart_df[chart_df['Keyword'] == keyword]
            if not keyword_data.empty:
                fig.add_trace(go.Bar(
                    name=f"{keyword} - {match_type}",
                    x=keyword_data['Transcript'],
                    y=keyword_data[match_type],
                    marker_color=colors[match_type],
                    legendgroup=keyword,
                    legendgrouptitle_text=keyword,
                    showlegend=True
                ))
    
    fig.update_layout(
        title='Match Type Breakdown by Transcript and Keyword',
        xaxis_title='Transcripts',
        yaxis_title='Number of Matches',
        barmode='stack',
        height=500,
        legend=dict(groupclick="toggleitem")
    )
    
    return fig

def create_resolution_summary_chart(all_results):
    """Create overall resolution summary."""
    summary_data = {
        'Total Transcripts': len(all_results),
        'Resolving YES': sum(1 for data in all_results.values() if data['market_resolution']),
        'Resolving NO': sum(1 for data in all_results.values() if not data['market_resolution'])
    }
    
    fig = go.Figure(data=[
        go.Bar(
            x=['Total Transcripts', 'Resolving YES', 'Resolving NO'],
            y=[summary_data['Total Transcripts'], summary_data['Resolving YES'], summary_data['Resolving NO']],
            marker_color=['#87CEEB', '#44ff44', '#ff4444'],
            text=[summary_data['Total Transcripts'], summary_data['Resolving YES'], summary_data['Resolving NO']],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title='Market Resolution Summary',
        yaxis_title='Number of Transcripts',
        height=300
    )
    
    return fig

def export_market_results(comparison_df, all_results):
    """Export comprehensive market analysis results."""
    
    # Create summary sheet
    summary_data = []
    for transcript_name, data in all_results.items():
        summary_data.append({
            'Transcript': transcript_name,
            'Market_Resolution': 'YES' if data['market_resolution'] else 'NO',
            'Total_Keyword_Instances': data['total_keywords'],
            'Word_Count': data['word_count'],
            'Keyword_Density_per_1000': round((data['total_keywords'] / data['word_count']) * 1000, 2) if data['word_count'] > 0 else 0
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Combine with detailed comparison data
    export_data = {
        'Summary': summary_df.to_csv(index=False),
        'Detailed_Analysis': comparison_df.to_csv(index=False)
    }
    
    return export_data

# Part 3: Main Streamlit Application for Prediction Market Analysis


def main():
    st.set_page_config(
        page_title="Prediction Market Resolution Tool",
        page_icon="🎯",
        layout="wide"
    )
    
    # Header with market context
    st.title("🎯 Prediction Market Resolution Tool")
    st.markdown("**Analyze transcripts for term detection according to prediction market rules**")
    
    # Market Rules Display
    with st.expander("📋 Market Resolution Rules", expanded=False):
        st.markdown("""
        ### Resolution Criteria:
        - **YES**: If the listed term is mentioned by anyone during the event
        - **NO**: If the term is not mentioned
        
        ### What Counts:
        ✅ **Any usage regardless of context**  
        ✅ **Pluralization/possessive forms** (e.g., "joy" → "joys", "joy's")  
        ✅ **Compound words** (e.g., "joy" in "killjoy")  
        ❌ **Other forms do NOT count** (e.g., "joyful" for "joy")
        

        """)
    
    # Sidebar for settings
    st.sidebar.header("⚙️ Analysis Settings")
    
    # Prediction market specific settings
    st.sidebar.subheader("🎯 Market Rules")
    case_sensitive = st.sidebar.checkbox("Case sensitive matching", value=False, 
                                       help="Whether to distinguish between 'Joy' and 'joy'")
    include_plurals = st.sidebar.checkbox("Include plurals/possessive", value=True,
                                        help="Include forms like 'joys', 'joy's'")
    include_compounds = st.sidebar.checkbox("Include compound words", value=True,
                                          help="Include words containing the term like 'killjoy'")
    
    # Analysis mode selection
    analysis_mode = st.sidebar.radio(
        "Analysis Mode",
        ["Single Event", "Multiple Events"],
        help="Analyze one event or compare multiple events"
    )
    
    # Initialize session state
    if 'transcripts' not in st.session_state:
        st.session_state.transcripts = {}
    if 'market_terms' not in st.session_state:
        st.session_state.market_terms = ""
    
    # Input section
    st.header("📝 Event Data")
    
    if analysis_mode == "Single Event":
        # Single event mode
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🎤 Event Transcript")
            
            uploaded_file = st.file_uploader(
                "Upload event transcript", 
                type=['txt', 'csv'],
                help="Upload transcript file from the event"
            )
            
            if uploaded_file:
                transcripts = load_transcripts_from_files([uploaded_file])
                if transcripts:
                    st.session_state.transcripts = transcripts
            
            # Text input
            transcript_text = st.text_area(
                "Or paste transcript text",
                height=200,
                placeholder="Paste the event transcript here...",
                value=list(st.session_state.transcripts.values())[0] if st.session_state.transcripts else ""
            )
            
            if transcript_text:
                st.session_state.transcripts = {"Event Transcript": transcript_text}
        
        with col2:
            st.subheader("🎯 Market Terms")
            
            market_terms = st.text_input(
                "Terms to analyze (comma-separated)",
                placeholder="term1, term2, term3",
                value=st.session_state.market_terms,
                help="Enter the exact terms from your prediction markets"
            )
            st.session_state.market_terms = market_terms
            
            # Show current settings summary
            if st.session_state.market_terms:
                st.info(f"""
                **Analysis Settings:**
                - Terms: {st.session_state.market_terms}
                - Case sensitive: {case_sensitive}
                - Include plurals: {include_plurals}
                - Include compounds: {include_compounds}
                """)
    
    else:
        # Multiple events mode
        st.subheader("📁 Upload Multiple Event Transcripts")
        
        uploaded_files = st.file_uploader(
            "Upload event transcript files",
            type=['txt', 'csv'],
            accept_multiple_files=True,
            help="Upload multiple transcript files for comparison"
        )
        
        if uploaded_files:
            transcripts = load_transcripts_from_files(uploaded_files)
            st.session_state.transcripts = transcripts
            
            if transcripts:
                st.success(f"Loaded {len(transcripts)} event transcript(s)")
                with st.expander("📋 Preview loaded events"):
                    for name, content in transcripts.items():
                        st.text(f"**{name}:** {len(content)} characters, {len(content.split())} words")
        
        # Manual input for multiple events
        st.subheader("✏️ Add Events Manually")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            event_name = st.text_input("Event Name", placeholder="e.g., Presidential Debate #1")
            event_transcript = st.text_area("Event Transcript", height=150, placeholder="Paste event transcript...")
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Add Event") and event_name and event_transcript:
                st.session_state.transcripts[event_name] = event_transcript
                st.success(f"Added '{event_name}'")
                st.rerun()
            
            if st.button("Clear All Events") and st.session_state.transcripts:
                st.session_state.transcripts = {}
                st.success("Cleared all events")
                st.rerun()
        
        # Show current events
        if st.session_state.transcripts:
            st.subheader("📋 Current Events")
            events_df = pd.DataFrame([
                {
                    'Event': name,
                    'Characters': len(content),
                    'Words': len(content.split()),
                    'Preview': content[:100] + '...' if len(content) > 100 else content
                }
                for name, content in st.session_state.transcripts.items()
            ])
            st.dataframe(events_df, use_container_width=True)
        
        # Terms input
        market_terms = st.text_input(
            "Market terms to analyze (comma-separated)",
            placeholder="term1, term2, term3",
            value=st.session_state.market_terms,
            help="Enter the exact terms from your prediction markets"
        )
        st.session_state.market_terms = market_terms
    
    # Analysis section
    if st.session_state.transcripts and st.session_state.market_terms:
        st.header("🔍 Market Resolution Analysis")
        
        # Process transcripts with market rules
        all_results = process_multiple_transcripts_market(
            st.session_state.transcripts,
            st.session_state.market_terms,
            case_sensitive,
            include_plurals,
            include_compounds
        )
        
        if analysis_mode == "Single Event":
            # Single event analysis
            event_name = list(all_results.keys())[0]
            event_data = all_results[event_name]
            results = event_data['results']
            
            # Market Resolution Summary
            st.subheader("🎯 Market Resolution")
            overall_resolution = 'YES' if event_data['market_resolution'] else 'NO'
            
            if overall_resolution == 'YES':
                st.success(f"**MARKET RESOLVES: {overall_resolution}** ✅")
            else:
                st.error(f"**MARKET RESOLVES: {overall_resolution}** ❌")
            
            # Individual term results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Matches", event_data['total_keywords'])
            with col2:
                resolving_terms = sum(1 for r in results if r['market_resolution'] == 'YES')
                st.metric("Terms Found", resolving_terms)
            with col3:
                st.metric("Terms Searched", len(results))
            
            # Detailed results table
            st.subheader("📊 Detailed Analysis")
            
            detailed_results = []
            for result in results:
                match_types = result['match_types']
                detailed_results.append({
                    'Term': result['keyword'],
                    'Market Resolution': result['market_resolution'],
                    'Total Matches': result['count'],
                    'Exact': len(match_types['exact']),
                    'Plural/Possessive': len(match_types['plural']) + len(match_types['possessive']),
                    'Compound Words': len(match_types['compound']),
                    'Density (per 1000 words)': result['density']
                })
            
            detailed_df = pd.DataFrame(detailed_results)
            st.dataframe(detailed_df, use_container_width=True)
            
            # Visualizations for single event
            if sum(detailed_df['Total Matches']) > 0:
                st.subheader("📊 Match Type Breakdown")
                
                # Create breakdown chart
                fig_breakdown = go.Figure()
                
                categories = ['Exact', 'Plural/Possessive', 'Compound Words']
                colors = ['#FFB6C1', '#98FB98', '#DDA0DD']
                
                for i, category in enumerate(categories):
                    fig_breakdown.add_trace(go.Bar(
                        name=category,
                        x=detailed_df['Term'],
                        y=detailed_df[category],
                        marker_color=colors[i],
                        text=detailed_df[category],
                        textposition='auto'
                    ))
                
                fig_breakdown.update_layout(
                    title='Match Types by Term',
                    xaxis_title='Terms',
                    yaxis_title='Number of Matches',
                    barmode='stack',
                    height=400
                )
                
                st.plotly_chart(fig_breakdown, use_container_width=True)
            
            # Highlighted transcript
            st.subheader("🎯 Highlighted Transcript")
            show_highlight = st.checkbox("Show highlighted text", value=True)
            
            if show_highlight and results:
                # Add legend for highlight colors
                st.markdown("""
                **Highlight Legend:**  
                🟣 **Exact matches** • 🟢 **Plurals** • 🔵 **Possessives** • 🟪 **Compounds**
                """)
                
                transcript_text = list(st.session_state.transcripts.values())[0]
                highlighted = highlight_text_with_types(transcript_text, results)
                
                st.markdown(
                    f'<div style="max-height: 500px; overflow-y: auto; padding: 15px; background-color: #f8f9fa; border-radius: 8px; border: 1px solid #dee2e6;">{highlighted}</div>',
                    unsafe_allow_html=True
                )
        
        else:
            # Multiple events comparison
            comparison_df = create_market_comparison_dataframe(all_results, st.session_state.market_terms)
            market_summary_df = create_market_resolution_summary(all_results)
            
            # Overall market summary
            st.subheader("🎯 Market Resolution Summary")
            
            yes_count = market_summary_df['Market Resolution'].value_counts().get('YES', 0)
            no_count = market_summary_df['Market Resolution'].value_counts().get('NO', 0)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Events", len(all_results))
            with col2:
                st.metric("Resolving YES", yes_count, delta=None)
            with col3:
                st.metric("Resolving NO", no_count, delta=None)
            with col4:
                success_rate = (yes_count / len(all_results)) * 100 if all_results else 0
                st.metric("Success Rate", f"{success_rate:.1f}%")
            
            # Market resolution summary table
            st.dataframe(market_summary_df, use_container_width=True)
            
            # Export functionality
            export_data = export_market_results(comparison_df, all_results)
            st.download_button(
                label="📥 Download Complete Analysis",
                data=export_data['Detailed_Analysis'],
                file_name="market_resolution_analysis.csv",
                mime="text/csv"
            )
            
            # Visualizations
            st.subheader("📊 Market Resolution Visualizations")
            
            # Resolution summary chart
            resolution_summary_fig = create_resolution_summary_chart(all_results)
            st.plotly_chart(resolution_summary_fig, use_container_width=True)
            
            # Resolution heatmap
            resolution_heatmap = create_market_resolution_chart(comparison_df, st.session_state.market_terms)
            if resolution_heatmap:
                st.plotly_chart(resolution_heatmap, use_container_width=True)
            
            # Match type breakdown
            match_breakdown_fig = create_match_type_breakdown_chart(comparison_df, st.session_state.market_terms)
            if match_breakdown_fig:
                st.plotly_chart(match_breakdown_fig, use_container_width=True)
            
            # Individual event details
            st.subheader("🔍 Individual Event Analysis")
            selected_event = st.selectbox(
                "Select event to analyze:",
                list(st.session_state.transcripts.keys())
            )
            
            if selected_event:
                event_results = all_results[selected_event]['results']
                event_resolution = 'YES' if all_results[selected_event]['market_resolution'] else 'NO'
                
                if event_resolution == 'YES':
                    st.success(f"**{selected_event} - RESOLVES: {event_resolution}** ✅")
                else:
                    st.error(f"**{selected_event} - RESOLVES: {event_resolution}** ❌")
                
                # Individual event details table
                individual_results = []
                for result in event_results:
                    match_types = result['match_types']
                    individual_results.append({
                        'Term': result['keyword'],
                        'Resolution': result['market_resolution'],
                        'Total': result['count'],
                        'Exact': len(match_types['exact']),
                        'Plural/Poss.': len(match_types['plural']) + len(match_types['possessive']),
                        'Compound': len(match_types['compound']),
                        'Density': result['density']
                    })
                
                individual_df = pd.DataFrame(individual_results)
                st.dataframe(individual_df, use_container_width=True)
                
                # Show highlighted text for selected event
                if st.checkbox(f"Show highlighted text for {selected_event}"):
                    st.markdown("""
                    **Highlight Legend:**  
                    🟣 **Exact** • 🟢 **Plurals** • 🔵 **Possessives** • 🟪 **Compounds**
                    """)
                    
                    highlighted = highlight_text_with_types(st.session_state.transcripts[selected_event], event_results)
                    st.markdown(
                        f'<div style="max-height: 400px; overflow-y: auto; padding: 15px; background-color: #f8f9fa; border-radius: 8px; border: 1px solid #dee2e6;">{highlighted}</div>',
                        unsafe_allow_html=True
                    )
    
    elif st.session_state.market_terms and not st.session_state.transcripts:
        st.info("📁 Please add event transcripts to analyze.")
    elif st.session_state.transcripts and not st.session_state.market_terms:
        st.info("🎯 Please enter market terms to search for.")
    else:
        st.info("📝 Please add event transcripts and market terms to begin analysis.")
    
    # Footer
    st.markdown("---")
    st.markdown("🎯 **Prediction Market Resolution Tool** • Built for accurate term detection according to market rules")

if __name__ == "__main__":
    main()

import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import platform
def set_font():
    system_name = platform.system()
    if system_name == "Windows":
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    elif system_name == "Darwin":
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC']
    else:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei']
    plt.rcParams['axes.unicode_minus'] = False 
plt.style.use('seaborn-v0_8')
plt.switch_backend('Agg')
set_font() 
def process_and_analyze(file_obj):    
    if file_obj is None:
        return [None] * 9
    try:
        df = pd.read_csv(file_obj.name, encoding='big5')
    except:
        try:
            df = pd.read_csv(file_obj.name, encoding='utf-8')
        except Exception as e:
            raise gr.Error(f"讀取失敗: {str(e)}")
    if 'dp002_timestamp' in df.columns:
        df['dt_timestamp'] = pd.to_datetime(df['dp002_timestamp'], errors='coerce')
        df['hour'] = df['dt_timestamp'].dt.hour
        df['weekday'] = df['dt_timestamp'].dt.day_name()
    numeric_cols = ['dp001_review_finish_rate', 'dp001_prac_score_rate', 'dp001_prac_during_time']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df_full = df.copy()
    df_clean = df.dropna(subset=['dp002_object_definition_name_zh-TW', 'dp001_record_plus_view_action'])
    fig_time = plt.figure(figsize=(10, 5))
    if 'hour' in df_clean.columns:
        hourly_counts = df_clean['hour'].value_counts().sort_index()
        hourly_counts.plot(kind='bar', color='#4c72b0', width=0.8)
        plt.title('Peak Learning Hours (活躍時段)', fontsize=14)
        plt.xlabel('Hour', fontsize=12)
        plt.xticks(rotation=0)
    plt.tight_layout()
    fig_action = plt.figure(figsize=(10, 5))
    if 'dp001_record_plus_view_action' in df_clean.columns:
        action_counts = df_clean['dp001_record_plus_view_action'].value_counts().head(10)
        action_counts.plot(kind='barh', color='#dd8452')
        plt.title('Top Learning Actions (熱門操作行為)', fontsize=14)
        plt.gca().invert_yaxis()
    plt.tight_layout()
    fig_cluster = plt.figure(figsize=(10, 6))
    if 'PseudoID' in df_clean.columns:
        student_features = df_clean.groupby('PseudoID').agg({
            'dp001_review_finish_rate': 'mean',
            'dp001_prac_score_rate': 'mean',
            'dp002_verb_id': 'count'
        }).dropna()
        if len(student_features) > 3:
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(student_features)
            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(scaled_features)
            scatter = plt.scatter(
                student_features['dp001_review_finish_rate'], 
                student_features['dp001_prac_score_rate'], 
                c=clusters, 
                cmap='viridis', 
                alpha=0.7,
                s=student_features['dp002_verb_id'] / 2
            )
            plt.title('Student Clustering (學生學習分群)', fontsize=14)
            plt.xlabel('Video Finish Rate', fontsize=12)
            plt.ylabel('Practice Score', fontsize=12)
            plt.colorbar(scatter, label='Cluster')
    plt.tight_layout()
    fig_corr = plt.figure(figsize=(8, 6))
    if 'PseudoID' in df_clean.columns and len(student_features) > 0:
        corr_matrix = student_features.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Behavior Correlation (行為變數相關性)', fontsize=14)
    plt.tight_layout()
    fig_week = plt.figure(figsize=(10, 5))
    if 'weekday' in df_full.columns:
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        df_full['weekday'] = pd.Categorical(df_full['weekday'], categories=days_order, ordered=True)
        weekly_counts = df_full['weekday'].value_counts().sort_index()
        sns.barplot(x=weekly_counts.index, y=weekly_counts.values, palette="Blues_d")
        plt.title('Weekly Learning Activity (每週學習活躍度)', fontsize=14)
        plt.ylabel('Activity Count')
        plt.xticks(rotation=45)
    plt.tight_layout()
    fig_verb = plt.figure(figsize=(10, 6))
    if 'dp002_verb_display_zh_TW' in df_full.columns:
        verb_counts = df_full['dp002_verb_display_zh_TW'].value_counts().head(6)
        if not verb_counts.empty:
            total = verb_counts.sum()
            verb_pct = (verb_counts / total * 100).round(1)
            ax = sns.barplot(x=verb_counts.values, y=verb_counts.index, palette='pastel')
            plt.title('Action Verb Distribution (主要學習動作分布 - Top 6)', fontsize=14)
            plt.xlabel('Count (次數)')
            plt.ylabel('Action Verb (動作)')
            for i, p in enumerate(ax.patches):
                width = p.get_width()
                ax.text(width + (width * 0.01),      
                        p.get_y() + p.get_height()/2, 
                        f'{int(width)} ({verb_pct.iloc[i]}%)', 
                        ha='left',                    
                        va='center',                  
                        fontsize=10,
                        color='black')
            plt.xlim(right=total * 1.15) 
    plt.tight_layout()
    fig_hist = plt.figure(figsize=(10, 5))
    if 'dp001_review_finish_rate' in df_full.columns:
        data_hist = df_full['dp001_review_finish_rate'].dropna()
        sns.histplot(data_hist, bins=20, kde=True, color='purple')
        plt.title('Video Completion Rate Distribution (影片完成率分布)', fontsize=14)
        plt.xlabel('Finish Rate (%)')
    plt.tight_layout()
    fig_prac = plt.figure(figsize=(10, 6))
    if 'dp001_prac_during_time' in df_full.columns and 'dp001_prac_score_rate' in df_full.columns:
        mask = (df_full['dp001_prac_during_time'] < 3600) & (df_full['dp001_prac_during_time'] > 0)
        subset = df_full[mask].dropna(subset=['dp001_prac_score_rate'])
        if not subset.empty:
            sns.scatterplot(data=subset, x='dp001_prac_during_time', y='dp001_prac_score_rate', alpha=0.5, color='teal')
            plt.title('Practice Duration vs Score (練習時長與成績關係)', fontsize=14)
            plt.xlabel('Duration (Seconds)')
            plt.ylabel('Score Rate')
    plt.tight_layout()
    fig_top_std = plt.figure(figsize=(10, 6))
    if 'PseudoID' in df_full.columns:
        top_students = df_full['PseudoID'].value_counts().head(15)
        sns.barplot(x=top_students.index.astype(str), y=top_students.values, palette="magma")
        plt.title('Top 15 Most Engaged Students (高參與度學生)', fontsize=14)
        plt.xlabel('Student ID')
        plt.ylabel('Interaction Count')
        plt.xticks(rotation=45)
    plt.tight_layout()
    return fig_time, fig_action, fig_cluster, fig_corr, fig_week, fig_verb, fig_hist, fig_prac, fig_top_std
with gr.Blocks(title="學生學習行為深度分析系統 V2.1") as demo:
    gr.Markdown("# 🎓 數位學習行為深度分析系統")
    with gr.Row():
        file_input = gr.File(label="上傳 CSV", file_types=[".csv"])
        analyze_btn = gr.Button("開始分析", variant="primary")
    with gr.Tabs():
        with gr.TabItem("📊 基礎分析"):
            with gr.Row():
                plot_time = gr.Plot(label="時間分析")
                plot_action = gr.Plot(label="行為序列")
            with gr.Row():
                plot_cluster = gr.Plot(label="學生分群")
            with gr.Row():
                plot_corr = gr.Plot(label="關聯分析")
        with gr.TabItem("📈 進階洞察"):
            with gr.Row():
                plot_week = gr.Plot(label="每週模式")
                plot_verb = gr.Plot(label="動作分布")
            with gr.Row():
                plot_hist = gr.Plot(label="影片完成率")
                plot_prac = gr.Plot(label="練習與成績")
            plot_top_std = gr.Plot(label="高參與學生")
    analyze_btn.click(
        fn=process_and_analyze, 
        inputs=file_input, 
        outputs=[plot_time, plot_action, plot_cluster, plot_corr, 
                 plot_week, plot_verb, plot_hist, plot_prac, plot_top_std]
    )
if __name__ == "__main__":

    demo.launch()

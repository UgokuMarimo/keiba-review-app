import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

def load_analysis_data(db_path):
    """
    Load prediction and payout data from the database for analysis.
    """
    with sqlite3.connect(db_path) as conn:
        query = """
        SELECT 
            p.race_id, p.race_number, p.umaban, p.horse_name,
            p.kaisai_date, p.keibajo, p.track_type, 
            p.pred_win, p.pred_rank, 
            p.result_rank, p.tansho_odds,
            pay.tansho_payout, pay.tansho_numbers
        FROM predictions p
        LEFT JOIN payouts pay ON p.race_id = pay.race_id
        WHERE p.result_rank IS NOT NULL
        """
        df = pd.read_sql_query(query, conn)
    
    if df.empty:
        return df

    # Convert kaisai_date to datetime
    df['kaisai_date'] = pd.to_datetime(df['kaisai_date'])
    df['month'] = df['kaisai_date'].dt.strftime('%Y-%m')
    
    # Calculate Expected Value (EV)
    # EV = Predicted Win Probability * Odds
    df['expected_value'] = df['pred_win'] * df['tansho_odds']
    
    # Determine if the prediction was a hit (Rank 1 prediction won)
    # Note: This is a simple definition of "Hit" for the model's top pick.
    # For betting simulation, we might look at whether we bought it.
    # Here we assume we bet on the Rank 1 horse.
    df['is_hit'] = (df['pred_rank'] == 1) & (df['result_rank'] == 1)
    
    # Calculate Return for Rank 1 bets
    # If pred_rank == 1 and it won, return is tansho_payout. Else 0.
    # We assume a flat bet of 100 yen for calculation simplicity in rates.
    def calculate_return(row):
        if row['pred_rank'] == 1:
            if row['result_rank'] == 1:
                return row['tansho_payout']
            else:
                return 0
        return 0

    df['return'] = df.apply(calculate_return, axis=1)
    
    return df

def calculate_metrics(df):
    """
    Calculate overall metrics based on Rank 1 predictions.
    """
    # Filter for only the top-ranked predictions (1 per race)
    # Assuming we bet on the horse predicted as Rank 1
    rank1_df = df[df['pred_rank'] == 1].copy()
    
    total_races = len(rank1_df)
    if total_races == 0:
        return 0, 0, 0, 0
        
    hit_count = rank1_df['is_hit'].sum()
    hit_rate = (hit_count / total_races) * 100
    
    total_investment = total_races * 100
    total_return = rank1_df['return'].sum()
    recovery_rate = (total_return / total_investment) * 100
    
    return total_races, hit_rate, recovery_rate, total_return

def plot_monthly_stats(df):
    """
    Plot monthly Hit Rate and Recovery Rate.
    """
    rank1_df = df[df['pred_rank'] == 1].copy()
    
    monthly_stats = rank1_df.groupby('month').apply(
        lambda x: pd.Series({
            'hit_rate': (x['is_hit'].sum() / len(x)) * 100,
            'recovery_rate': (x['return'].sum() / (len(x) * 100)) * 100,
            'count': len(x)
        })
    ).reset_index()
    
    fig = go.Figure()
    
    # Hit Rate Bar
    fig.add_trace(go.Bar(
        x=monthly_stats['month'],
        y=monthly_stats['hit_rate'],
        name='的中率 (%)',
        yaxis='y1',
        marker_color='rgba(55, 83, 109, 0.7)'
    ))
    
    # Recovery Rate Line
    fig.add_trace(go.Scatter(
        x=monthly_stats['month'],
        y=monthly_stats['recovery_rate'],
        name='回収率 (%)',
        yaxis='y2',
        mode='lines+markers',
        line=dict(color='rgb(219, 64, 82)', width=3)
    ))
    
    fig.update_layout(
        title='月別 的中率・回収率推移',
        xaxis=dict(title='年月'),
        yaxis=dict(title='的中率 (%)', side='left', range=[0, 100]),
        yaxis2=dict(title='回収率 (%)', side='right', overlaying='y', showgrid=False),
        legend=dict(x=0.01, y=0.99),
        hovermode='x unified'
    )
    
    return fig

def plot_venue_stats(df):
    """
    Plot Hit Rate and Recovery Rate by Venue (Keibajo).
    """
    rank1_df = df[df['pred_rank'] == 1].copy()
    
    venue_stats = rank1_df.groupby('keibajo').apply(
        lambda x: pd.Series({
            'hit_rate': (x['is_hit'].sum() / len(x)) * 100,
            'recovery_rate': (x['return'].sum() / (len(x) * 100)) * 100,
            'count': len(x)
        })
    ).reset_index().sort_values('recovery_rate', ascending=False)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=venue_stats['keibajo'],
        y=venue_stats['hit_rate'],
        name='的中率 (%)',
        marker_color='rgba(55, 83, 109, 0.7)'
    ))
    
    fig.add_trace(go.Bar(
        x=venue_stats['keibajo'],
        y=venue_stats['recovery_rate'],
        name='回収率 (%)',
        marker_color='rgba(26, 118, 255, 0.7)'
    ))
    
    # Add a line for 100% recovery
    fig.add_shape(
        type="line",
        x0=-0.5, y0=100, x1=len(venue_stats)-0.5, y1=100,
        line=dict(color="red", width=2, dash="dash"),
    )
    
    fig.update_layout(
        title='競馬場別 成績 (回収率順)',
        xaxis=dict(title='競馬場'),
        yaxis=dict(title='パーセント (%)'),
        barmode='group',
        legend=dict(x=0.01, y=0.99)
    )
    
    return fig

def plot_track_type_stats(df):
    """
    Plot stats by Track Type (Turf/Dirt).
    """
    rank1_df = df[df['pred_rank'] == 1].copy()
    
    # Simple normalization of track types if needed (e.g. '芝' vs '芝右')
    # Assuming '芝' and 'ダート' are contained in the string
    def normalize_track(t):
        if '芝' in str(t): return '芝'
        if 'ダ' in str(t): return 'ダート'
        return 'その他'
        
    rank1_df['simple_track'] = rank1_df['track_type'].apply(normalize_track)
    
    track_stats = rank1_df.groupby('simple_track').apply(
        lambda x: pd.Series({
            'hit_rate': (x['is_hit'].sum() / len(x)) * 100,
            'recovery_rate': (x['return'].sum() / (len(x) * 100)) * 100,
            'count': len(x)
        })
    ).reset_index()
    
    fig = px.bar(
        track_stats, 
        x='simple_track', 
        y=['hit_rate', 'recovery_rate'],
        barmode='group',
        title='芝・ダート別 成績',
        labels={'value': 'パーセント (%)', 'simple_track': 'コース区分', 'variable': '指標'}
    )
    
    # Add 100% line
    fig.add_shape(
        type="line",
        x0=-0.5, y0=100, x1=len(track_stats)-0.5, y1=100,
        line=dict(color="red", width=2, dash="dash"),
    )
    
    return fig

def plot_ev_analysis(df):
    """
    Analyze performance based on Expected Value bins.
    """
    # Use all horses, not just rank 1, to see if high EV horses perform well
    # But for "Accuracy Analysis" usually we care about our predictions.
    # Let's stick to Rank 1 predictions for now to see "If we bought the AI's top pick, how does EV correlate?"
    # OR we can look at "All horses with EV > 1.0"
    
    # Let's look at Rank 1 predictions first as that's the primary output
    rank1_df = df[df['pred_rank'] == 1].copy()
    
    # Create bins for Expected Value
    bins = [0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 5.0, 100]
    labels = ['<0.5', '0.5-0.8', '0.8-1.0', '1.0-1.2', '1.2-1.5', '1.5-2.0', '2.0-5.0', '>5.0']
    
    rank1_df['ev_bin'] = pd.cut(rank1_df['expected_value'], bins=bins, labels=labels)
    
    ev_stats = rank1_df.groupby('ev_bin', observed=True).apply(
        lambda x: pd.Series({
            'hit_rate': (x['is_hit'].sum() / len(x)) * 100 if len(x) > 0 else 0,
            'recovery_rate': (x['return'].sum() / (len(x) * 100)) * 100 if len(x) > 0 else 0,
            'count': len(x)
        })
    ).reset_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=ev_stats['ev_bin'],
        y=ev_stats['hit_rate'],
        name='的中率 (%)',
        marker_color='rgba(55, 83, 109, 0.7)'
    ))
    
    fig.add_trace(go.Bar(
        x=ev_stats['ev_bin'],
        y=ev_stats['recovery_rate'],
        name='回収率 (%)',
        marker_color='rgba(26, 118, 255, 0.7)'
    ))
    
    # Add count as text on bars or separate trace?
    # Let's add text to recovery rate bars
    fig.update_traces(texttemplate='%{y:.1f}', textposition='outside')
    
    fig.add_shape(
        type="line",
        x0=-0.5, y0=100, x1=len(ev_stats)-0.5, y1=100,
        line=dict(color="red", width=2, dash="dash"),
    )
    
    fig.update_layout(
        title='期待値(EV)別 成績 (AI予測1位の馬)',
        xaxis=dict(title='期待値 (予測勝率 × オッズ)'),
        yaxis=dict(title='パーセント (%)'),
        barmode='group',
        legend=dict(x=0.01, y=0.99)
    )
    
    return fig

def render_analysis_page(db_path):
    st.header("📊 予測精度分析")
    
    df = load_analysis_data(db_path)
    
    if df.empty:
        st.info("分析対象のデータがまだありません。")
        return

    # --- Overall Metrics ---
    total_races, hit_rate, recovery_rate, total_return = calculate_metrics(df)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("総予測レース数", f"{total_races} レース")
    col2.metric("単勝的中率 (1位)", f"{hit_rate:.1f}%")
    col3.metric("単勝回収率 (1位)", f"{recovery_rate:.1f}%", delta=f"{recovery_rate - 100:.1f}%")
    col4.metric("総払戻金", f"{int(total_return):,} 円")
    
    st.markdown("---")
    
    # --- Tabs for different analyses ---
    tab1, tab2, tab3, tab4 = st.tabs(["📈 時系列・全体", "🏟️ 会場・コース別", "💰 期待値分析", "📋 的中レース一覧"])
    
    with tab1:
        st.subheader("月別推移")
        st.plotly_chart(plot_monthly_stats(df), use_container_width=True)
        
        # Cumulative Recovery Rate
        rank1_df = df[df['pred_rank'] == 1].sort_values('kaisai_date').copy()
        rank1_df['cumulative_return'] = rank1_df['return'].cumsum()
        rank1_df['cumulative_invest'] = (rank1_df.index.to_series().reset_index(drop=True).index + 1) * 100
        rank1_df['cumulative_recovery'] = (rank1_df['cumulative_return'] / rank1_df['cumulative_invest']) * 100
        
        fig_cum = px.line(rank1_df, x='kaisai_date', y='cumulative_recovery', title='累積回収率の推移')
        fig_cum.add_hline(y=100, line_dash="dash", line_color="red")
        st.plotly_chart(fig_cum, use_container_width=True)

    with tab2:
        col_venue, col_track = st.columns(2)
        with col_venue:
            st.plotly_chart(plot_venue_stats(df), use_container_width=True)
        with col_track:
            st.plotly_chart(plot_track_type_stats(df), use_container_width=True)

    with tab3:
        st.subheader("期待値 (Expected Value) 分析")
        st.markdown("""
        **期待値 (EV)** = AI予測勝率 × 単勝オッズ
        
        - **EV > 1.0**: 理論上、長期的にプラスになる賭け
        - このグラフは、AIが1位と予測した馬の期待値ごとの成績を示しています。
        """)
        st.plotly_chart(plot_ev_analysis(df), use_container_width=True)
        
        # Simulation
        st.subheader("期待値ベースのシミュレーション")
        ev_threshold = st.slider("期待値の閾値 (これ以上の馬のみ購入)", 0.5, 2.0, 1.0, 0.1)
        
        sim_df = df[(df['pred_rank'] == 1) & (df['expected_value'] >= ev_threshold)]
        if not sim_df.empty:
            sim_races = len(sim_df)
            sim_hits = sim_df['is_hit'].sum()
            sim_return = sim_df['return'].sum()
            sim_invest = sim_races * 100
            
            s_col1, s_col2, s_col3 = st.columns(3)
            s_col1.metric("購入レース数", f"{sim_races} / {total_races}")
            s_col2.metric("的中率", f"{(sim_hits/sim_races)*100:.1f}%")
            rec_rate = (sim_return/sim_invest)*100
            s_col3.metric("回収率", f"{rec_rate:.1f}%", delta=f"{rec_rate-100:.1f}%")
        else:
            st.warning(f"期待値 {ev_threshold} 以上の予測馬はいませんでした。")

    with tab4:
        st.subheader("的中レース一覧")
        hit_df = df[(df['pred_rank'] == 1) & (df['result_rank'] == 1)].copy()
        if not hit_df.empty:
            display_cols = ['kaisai_date', 'keibajo', 'race_number', 'horse_name', 'pred_win', 'tansho_odds', 'tansho_payout', 'expected_value']
            st.dataframe(
                hit_df[display_cols].sort_values('kaisai_date', ascending=False).style.format({
                    'pred_win': '{:.1%}',
                    'tansho_odds': '{:.1f}',
                    'expected_value': '{:.2f}'
                })
            )
        else:
            st.info("まだ的中したレースがありません。")

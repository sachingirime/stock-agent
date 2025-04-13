# import streamlit as st
# import plotly.graph_objects as go

# def run_dashboard(results, signals, df):
#     st.title("📊 Interactive Trading Strategy Dashboard")

#     # Align all data
#     df_aligned, results_aligned = df.align(results, join='inner', axis=0)
#     signals_aligned = signals.loc[results_aligned.index]

#     # ✅ Exit early if no data to visualize
#     if results_aligned.empty or df_aligned.empty:
#         st.warning("Not enough data to visualize. Try a different time range or interval.")
#         return

#     # ✅ Exit early if Close price is NaN or no rows exist
#     if 'Close' not in df_aligned.columns or df_aligned['Close'].isna().all():
#         st.warning("No valid Close prices found in data.")
#         return

#     fig = go.Figure()

#     fig.add_trace(go.Scatter(x=results_aligned.index, y=results_aligned['PortfolioValue'],
#                              mode='lines', name='Portfolio Value'))

#     try:
#         buy_hold = df_aligned['Close'] * (results_aligned['PortfolioValue'].iloc[0] / df_aligned['Close'].iloc[0])
#         fig.add_trace(go.Scatter(x=results_aligned.index, y=buy_hold,
#                                  mode='lines', name='Buy & Hold'))
#     except IndexError:
#         st.warning("Insufficient data to compute Buy & Hold baseline.")
#         return

#     buy_signals = signals_aligned[signals_aligned == 1].index
#     sell_signals = signals_aligned[signals_aligned == -1].index

#     fig.add_trace(go.Scatter(x=buy_signals, y=df_aligned.loc[buy_signals]['Close'],
#                              mode='markers', marker=dict(color='green', size=12, symbol='triangle-up'),
#                              name='Buy Signal'))

#     fig.add_trace(go.Scatter(x=sell_signals, y=df_aligned.loc[sell_signals]['Close'],
#                              mode='markers', marker=dict(color='red', size=12, symbol='triangle-down'),
#                              name='Sell Signal'))

#     fig.update_layout(
#         title="Portfolio vs. Buy & Hold with Trade Signals",
#         xaxis_title="Date",
#         yaxis_title="Value (USD)",
#         legend_title="Legend",
#         hovermode="x unified"
#     )

#     st.plotly_chart(fig, use_container_width=True)
#     st.subheader("Recent Signals")
#     st.dataframe(signals_aligned.tail(10))

# # visualize.py
# import streamlit as st
# import plotly.graph_objects as go

# def run_dashboard(df):
#     st.subheader("📈 RL Agent Trading Dashboard")
    
#     fig = go.Figure()

#     fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))

#     buy_signals = df[df['signals'] == 1]
#     sell_signals = df[df['signals'] == -1]

#     fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'], mode='markers', marker=dict(color='green', size=10), name='Buy'))
#     fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'], mode='markers', marker=dict(color='red', size=10), name='Sell'))

#     fig.update_layout(title="RL Agent Buy/Sell Signals", xaxis_title="Date", yaxis_title="Price")
#     st.plotly_chart(fig)

#     st.write("Recent signals")
#     st.dataframe(df[['Close', 'signals']].tail(10))



import streamlit as st
import plotly.graph_objects as go

def run_dashboard(*args):
    st.subheader("📈 Trading Strategy Dashboard")

    if len(args) == 3:
        # For Rule-Based or ML-Based strategy
        results, signals, df = args
        df_aligned, results_aligned = df.align(results, join='inner', axis=0)
        signals_aligned = signals.loc[results_aligned.index]

        if results_aligned.empty or df_aligned.empty:
            st.warning("Not enough data to visualize. Try a different time range or interval.")
            return

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results_aligned.index,
            y=results_aligned['PortfolioValue'],
            mode='lines',
            name='Portfolio Value'
        ))

        try:
            buy_hold = df_aligned['Close'] * (results_aligned['PortfolioValue'].iloc[0] / df_aligned['Close'].iloc[0])
            fig.add_trace(go.Scatter(
                x=results_aligned.index,
                y=buy_hold,
                mode='lines',
                name='Buy & Hold'
            ))
        except IndexError:
            st.warning("Insufficient data to compute Buy & Hold baseline.")
            return

        buy_signals = signals_aligned[signals_aligned == 1].index
        sell_signals = signals_aligned[signals_aligned == -1].index
        fig.add_trace(go.Scatter(
            x=buy_signals,
            y=df_aligned.loc[buy_signals]['Close'],
            mode='markers',
            marker=dict(color='green', size=12, symbol='triangle-up'),
            name='Buy Signal'
        ))
        fig.add_trace(go.Scatter(
            x=sell_signals,
            y=df_aligned.loc[sell_signals]['Close'],
            mode='markers',
            marker=dict(color='red', size=12, symbol='triangle-down'),
            name='Sell Signal'
        ))

        # Set dragmode to "pan" so click-drag moves the chart.
        # Also include a range slider and enable scroll-based zoom.
        fig.update_layout(
            title="Portfolio vs. Buy & Hold with Trade Signals",
            xaxis_title="Date",
            yaxis_title="Value (USD)",
            legend_title="Legend",
            hovermode="x unified",
            dragmode="pan",  # Enable panning on click-and-drag
            xaxis=dict(
    rangeslider=dict(visible=True),
    tickformat="%b %d\n%Y",       # Display on axis
    tickangle=0,
    hoverformat="%Y-%m-%d %H:%M"  # 🟢 Show full date/time on hover
)


        )

        # Enable scroll zoom and panning via the config
        st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})

        st.subheader("Recent Signals")
        st.dataframe(signals_aligned.tail(10))

    elif len(args) == 1:
        # For the case with a single DataFrame (e.g. RL Agent strategy)
        df = args[0]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Close'],
            mode='lines',
            name='Close Price'
        ))

        buy_signals = df[df['signals'] == 1]
        sell_signals = df[df['signals'] == -1]
        fig.add_trace(go.Scatter(
            x=buy_signals.index,
            y=buy_signals['Close'],
            mode='markers',
            marker=dict(color='green', size=10),
            name='Buy'
        ))
        fig.add_trace(go.Scatter(
            x=sell_signals.index,
            y=sell_signals['Close'],
            mode='markers',
            marker=dict(color='red', size=10),
            name='Sell'
        ))

        fig.update_layout(
            title="RL Agent Buy/Sell Signals",
            xaxis_title="Date",
            yaxis_title="Price",
            dragmode="pan",
            xaxis=dict(rangeslider=dict(visible=True))
        )

        st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
        st.subheader("Recent signals")
        st.dataframe(df[['Close', 'signals']].tail(10))

    else:
        st.error("Invalid arguments provided to run_dashboard.")

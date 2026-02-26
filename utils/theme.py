import streamlit as st
import plotly.graph_objects as go

class Colors:
    """Color palette for the dashboard"""
    # Primary colors
    PRUSSIAN_BLUE = "#0A1A2F"
    CHARCOAL = "#1E1E1E"
    SLATE_GRAY = "#5A6A7A"
    PLATINUM = "#F7F9FC"

    # Accent colors
    BLUE_ENERGY = "#3A8DFF"
    MINT_LEAF = "#4CC9A6"
    SUNSET_ORANGE = "#FFB84D"
    CORAL_RED = "#FF6B6B"

    # Chart colors
    CHART_COLORS = [
        "#3A8DFF",  # Blue
        "#4CC9A6",  # Mint
        "#FFB84D",  # Orange
        "#FF6B6B",  # Coral
        "#9B59B6",  # Purple
        "#3498DB",  # Light Blue
        "#E74C3C",  # Red
        "#F39C12",  # Gold
        "#1ABC9C",  # Turquoise
        "#34495E",  # Dark Gray
        "#C19AB7",  # Lilac
        "#F8FA90",  # Lime
        "#EDC9B2",  # Appricot
        "#BAFFDF",  # Aquamarine
        "#C1FFF2",  # Icy Aqua
        "#839788",  # Muted Teal
        "#F68EB0",  # Bubblegum
        "#9A955B",  # Leaf
        "#F4978E",  # Salmon
        "#8332AC"   # Indigo
    ]

class Components:
    """Reusable UI components"""
    @staticmethod
    def page_header(title:str) -> str:
        """Create a styled page header"""
        return f"""
        <div style='background: linear-gradient(135deg, #CC7A00 0%, #D91F1F 100%);
            padding: 2rem; border-radius: 10px; margin-bottom: 2rem;'>
            <h1 style='color: white; margin: 0; text-align: center; font-size: 2.5rem;'>{title}</h1>
        </div>
        """
    @staticmethod
    def section_header(title: str, icon: str = "") -> str:
        """Create a styled section header"""
        return f"""
        <div style='margin: 2rem 0 1rem 0; padding-bottom: 0.5rem; 
                    border-bottom: 2px solid {Colors.SUNSET_ORANGE};'>
            <h2 style='color: {Colors.PLATINUM}; margin: 0; font-size: 1.8rem;'>
                {icon} {title}
            </h2>
        </div>
        """
    
    @staticmethod
    def metric_card(title: str, value: str, delta: str = "",
                    delta_positive: bool = True, card_type: str = "primary") -> str:
        """Create a styled metric card"""
        # Color mapping
        colors = {
            "info": Colors.BLUE_ENERGY,
            "success": Colors.MINT_LEAF,
            "warning": Colors.SUNSET_ORANGE,
            "error": Colors.CORAL_RED

        }
        border_color = colors.get(card_type, Colors.BLUE_ENERGY)
        delta_color = Colors.MINT_LEAF if delta_positive else Colors.CORAL_RED
        
        delta_html = f"""
        <p style='color: {delta_color}; margin: 0.5rem 0 0 0; font-size: 0.85rem;'>
            {delta}
        </p>
        """ if delta else ""

        return f"""
        <div style='background-color: {Colors.CHARCOAL};
                    border: 1px solid {Colors.SLATE_GRAY};
                    border-top: 4px solid {border_color};
                    padding: 1rem; border-radius: 10px; height: 100%;
                    transition: transform 0.3s ease;'>
            <div style='display: flex; align-items: center; margin-bottom: 0.5rem;'>
                <p style='color: {Colors.SLATE_GRAY}; margin: 0; font-size: 0.85rem;
                            text-transform: uppercase; letter-spacing: 1px;'>
                    {title}
                </p>
            </div>
            <p style='color: {Colors.PLATINUM}; margin: 0; font-size: 1.6rem; font-weight: 700;'>
                {value}
            </p>
            {delta_html}
        </div>
        """
    @staticmethod
    def insight_box(title: str, content: str, box_type: str = "info", min_height: str = "auto") -> str:
        """Create a styled insight/info box with optional min-height"""
        # Color and icon mapping
        config = {
            "info": {"color": Colors.BLUE_ENERGY, "bg": "rgba(58, 141, 255, 0.15)"},
            "success": {"color": Colors.MINT_LEAF, "bg": "rgba(76, 201, 166, 0.15)"},
            "warning": {"color": Colors.SUNSET_ORANGE, "bg": "rgba(255, 184, 77, 0.15)"},
            "error": {"color": Colors.CORAL_RED, "bg": "rgba(255, 107, 107, 0.15)"}
        }
        style = config.get(box_type, config["info"])
        
        flex_style = "display: flex; flex-direction: column;" if min_height != "auto" else ""
        height_style = f"min-height: {min_height};" if min_height != "auto" else ""
        
        return f"""
        <div style='background-color: {style["bg"]}; 
                    padding: 1rem; border-radius: 8px; margin: 1rem 0;
                    border-left: 6px solid {style["color"]}; 
                    {height_style} {flex_style}'>
            <h4 style='color: {style["color"]}; margin: 0 0 0.5rem 0;'>{title}</h4>
            <div style='flex-grow: 1; color: {Colors.PLATINUM};'>{content}</div>
        </div>
        """


def apply_chart_theme(fig: go.Figure) -> go.Figure:
    """Apply consistent dark theme to Plotly charts"""
    
    fig.update_layout(
        paper_bgcolor=Colors.PRUSSIAN_BLUE,
        plot_bgcolor=Colors.PRUSSIAN_BLUE,
        font=dict(
            family="Roboto, sans-serif",
            size=12,
            color=Colors.PLATINUM
        ),
        title=dict(
            font=dict(size=16, color=Colors.PLATINUM),
            x=0,
            xanchor='left'
        ),
        legend=dict(
            bgcolor=Colors.CHARCOAL,
            bordercolor=Colors.SLATE_GRAY,
            borderwidth=1,
            font=dict(color=Colors.PLATINUM)
        ),
        xaxis=dict(
            gridcolor=Colors.SLATE_GRAY,
            gridwidth=0.5,
            showline=True,
            linecolor=Colors.SLATE_GRAY,
            color=Colors.PLATINUM
        ),
        yaxis=dict(
            gridcolor=Colors.SLATE_GRAY,
            gridwidth=0.5,
            showline=True,
            linecolor=Colors.SLATE_GRAY,
            color=Colors.PLATINUM
        ),
        hoverlabel=dict(
            bgcolor=Colors.CHARCOAL,
            font_size=12,
            font_family="Roboto, sans-serif",
            bordercolor=Colors.BLUE_ENERGY
        ),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig


def init_page(page_name: str, icon: str = "📊"):
    """Initialize page with common settings"""
    try:
        st.set_page_config(
            page_title=f"{page_name} | Retail Analytics",
            page_icon=icon,
            layout="wide",
            initial_sidebar_state="expanded"
        )
    except:
        # page config already set, skip
        pass
    
    # Load custom CSS
    try:
        with open('../style.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        try:
            with open('style.css') as f:
                st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        except FileNotFoundError:
            pass  # CSS not critical for functionality

    
def format_currency(value: float, currency: str = "£") -> str:
    """Format number as currency"""
    return f"{currency}{value:,.2f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format number as percentage"""
    return f"{value:.{decimals}f}%"


def format_number(value: float, decimals: int = 0) -> str:
    """Format number with thousands separator"""
    return f"{value:,.{decimals}f}"

"""
Chart generation tools for the ChartGenerator agent
"""

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import base64
import json
import re
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from datetime import datetime, timedelta
import logging

# Configure logger
logger = logging.getLogger(__name__)

# Chart configuration constants
CHART_CONFIG = {
    'width': 800,
    'height': 500,
    'dpi': 300,  # High resolution for PDF
    'format': 'png',
    'background': 'white',
    'font_family': 'Arial, sans-serif',
    'font_size': 12,
    'title_size': 16,
    'colors': {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e',
        'success': '#2ca02c',
        'warning': '#ffc107',
        'danger': '#d62728',
        'neutral': '#7f7f7f',
        'senior': '#1f77b4',
        'mid': '#ff7f0e',
        'junior': '#2ca02c'
    },
    'color_palette': [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
}

def apply_pdf_styling(fig):
    """Apply consistent styling for PDF output"""
    fig.update_layout(
        font_family=CHART_CONFIG['font_family'],
        font_size=CHART_CONFIG['font_size'],
        title_font_size=CHART_CONFIG['title_size'],
        plot_bgcolor=CHART_CONFIG['background'],
        paper_bgcolor=CHART_CONFIG['background'],
        margin=dict(l=60, r=60, t=80, b=60),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        width=CHART_CONFIG['width'],
        height=CHART_CONFIG['height']
    )
    return fig

def chart_to_base64(fig, width=None, height=None) -> str:
    """Convert Plotly figure to base64 for PDF embedding"""
    try:
        width = width or CHART_CONFIG['width']
        height = height or CHART_CONFIG['height']

        img_bytes = pio.to_image(
            fig,
            format=CHART_CONFIG['format'],
            width=width,
            height=height,
            scale=2  # High DPI for PDF
        )
        img_str = base64.b64encode(img_bytes).decode()
        return f'<img src="data:image/png;base64,{img_str}" style="width: 100%; max-width: {width}px; height: auto; margin: 10px 0;">'
    except Exception as e:
        logger.error(f"Failed to convert chart to base64: {e}")
        return f'<div style="border: 1px solid #ccc; padding: 20px; text-align: center; color: #666;">Chart generation failed: {str(e)}</div>'

def create_error_chart(error_message: str) -> str:
    """Create an error placeholder chart"""
    fig = go.Figure()
    fig.add_annotation(
        text=f"Chart Error: {error_message}",
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color="red")
    )
    fig.update_layout(
        title="Chart Generation Failed",
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False)
    )
    fig = apply_pdf_styling(fig)
    return chart_to_base64(fig)

def create_budget_pie_chart(data: Dict) -> str:
    """Create professional pie chart for budget breakdown"""
    try:
        if not validate_chart_data(data, ['categories', 'values']):
            return create_error_chart("Invalid budget data: missing categories or values")

        fig = px.pie(
            values=data['values'],
            names=data['categories'],
            title='Project Budget Breakdown',
            color_discrete_sequence=CHART_CONFIG['color_palette']
        )
        fig = apply_pdf_styling(fig)
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            showlegend=True
        )
        return chart_to_base64(fig)
    except Exception as e:
        logger.error(f"Failed to create budget pie chart: {e}")
        return create_error_chart(f"Budget chart error: {str(e)}")

def create_timeline_chart(data: Dict) -> str:
    """Create project timeline visualization"""
    try:
        if not validate_chart_data(data, ['months', 'costs']):
            return create_error_chart("Invalid timeline data: missing months or costs")

        fig = go.Figure()

        # Monthly costs
        fig.add_trace(go.Scatter(
            x=data['months'],
            y=data['costs'],
            mode='lines+markers',
            name='Monthly Cost',
            line=dict(color=CHART_CONFIG['colors']['primary'], width=3),
            marker=dict(size=8)
        ))

        # Cumulative costs if available
        if 'cumulative' in data and data['cumulative']:
            fig.add_trace(go.Scatter(
                x=data['months'],
                y=data['cumulative'],
                mode='lines+markers',
                name='Cumulative Cost',
                line=dict(color=CHART_CONFIG['colors']['secondary'], width=3, dash='dash'),
                marker=dict(size=8)
            ))

        fig.update_layout(
            title='Project Cost Timeline',
            xaxis_title='Timeline',
            yaxis_title='Cost ($)',
            yaxis=dict(tickformat='$,.0f')
        )
        fig = apply_pdf_styling(fig)
        return chart_to_base64(fig)
    except Exception as e:
        logger.error(f"Failed to create timeline chart: {e}")
        return create_error_chart(f"Timeline chart error: {str(e)}")

def create_resource_chart(data: Dict) -> str:
    """Create stacked bar chart for resource allocation"""
    try:
        if not validate_chart_data(data, ['roles']):
            return create_error_chart("Invalid resource data: missing roles")

        fig = go.Figure()

        levels = ['senior', 'mid', 'junior']
        colors = [CHART_CONFIG['colors'].get(level, CHART_CONFIG['colors']['primary']) for level in levels]

        for i, level in enumerate(levels):
            if level in data and data[level]:
                fig.add_trace(go.Bar(
                    name=level.title(),
                    x=data['roles'],
                    y=data[level],
                    marker_color=colors[i]
                ))

        fig.update_layout(
            title='Team Composition by Role',
            barmode='stack',
            xaxis_title='Roles',
            yaxis_title='Number of Resources'
        )
        fig = apply_pdf_styling(fig)
        return chart_to_base64(fig)
    except Exception as e:
        logger.error(f"Failed to create resource chart: {e}")
        return create_error_chart(f"Resource chart error: {str(e)}")

def create_roi_chart(data: Dict) -> str:
    """Create ROI projection chart"""
    try:
        if not validate_chart_data(data, ['quarters']):
            return create_error_chart("Invalid ROI data: missing quarters")

        fig = go.Figure()

        # Investment line
        if 'investment' in data:
            fig.add_trace(go.Scatter(
                x=data['quarters'],
                y=data['investment'],
                name='Investment',
                line=dict(color=CHART_CONFIG['colors']['danger'], width=3),
                marker=dict(size=8)
            ))

        # Returns line
        if 'returns' in data:
            fig.add_trace(go.Scatter(
                x=data['quarters'],
                y=data['returns'],
                name='Returns',
                line=dict(color=CHART_CONFIG['colors']['success'], width=3),
                marker=dict(size=8)
            ))

        # Net value line
        if 'net_value' in data:
            fig.add_trace(go.Scatter(
                x=data['quarters'],
                y=data['net_value'],
                name='Net Value',
                line=dict(color=CHART_CONFIG['colors']['primary'], width=3, dash='dash'),
                marker=dict(size=8)
            ))

        fig.update_layout(
            title='ROI Projection Over Time',
            xaxis_title='Quarter',
            yaxis_title='Value ($)',
            yaxis=dict(tickformat='$,.0f')
        )
        fig = apply_pdf_styling(fig)
        return chart_to_base64(fig)
    except Exception as e:
        logger.error(f"Failed to create ROI chart: {e}")
        return create_error_chart(f"ROI chart error: {str(e)}")

def create_risk_matrix_chart(data: Dict) -> str:
    """Create risk matrix scatter plot with jitter to prevent overlapping"""
    try:
        if not validate_chart_data(data, ['risks', 'probability', 'impact']):
            return create_error_chart("Invalid risk data: missing risks, probability, or impact")

        import random

        # Apply small random jitter to prevent overlapping points
        jitter_amount = 0.02  # Small offset to separate overlapping points
        adjusted_prob = []
        adjusted_impact = []

        for i, (prob, impact) in enumerate(zip(data['probability'], data['impact'])):
            # Add small random offset while keeping within bounds
            jitter_x = random.uniform(-jitter_amount, jitter_amount)
            jitter_y = random.uniform(-jitter_amount, jitter_amount)

            # Ensure values stay within [0, 1] bounds
            new_prob = max(0.01, min(0.99, prob + jitter_x))
            new_impact = max(0.01, min(0.99, impact + jitter_y))

            adjusted_prob.append(new_prob)
            adjusted_impact.append(new_impact)

        fig = px.scatter(
            x=adjusted_prob,
            y=adjusted_impact,
            size=data.get('sizes', [50] * len(data['risks'])),
            color=data['risks'],
            hover_name=data['risks'],
            title='Risk Assessment Matrix',
            labels={'x': 'Probability', 'y': 'Impact'},
            color_discrete_sequence=CHART_CONFIG['color_palette']
        )

        # Add quadrant lines
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=0.5, line_dash="dash", line_color="gray", opacity=0.5)

        # Add quadrant labels
        fig.add_annotation(x=0.25, y=0.75, text="Low Prob<br>High Impact", showarrow=False, font=dict(size=10, color="gray"))
        fig.add_annotation(x=0.75, y=0.75, text="High Prob<br>High Impact", showarrow=False, font=dict(size=10, color="gray"))
        fig.add_annotation(x=0.25, y=0.25, text="Low Prob<br>Low Impact", showarrow=False, font=dict(size=10, color="gray"))
        fig.add_annotation(x=0.75, y=0.25, text="High Prob<br>Low Impact", showarrow=False, font=dict(size=10, color="gray"))

        # Add text annotations for each risk to make them more visible
        for i, risk_name in enumerate(data['risks']):
            fig.add_annotation(
                x=adjusted_prob[i],
                y=adjusted_impact[i],
                text=risk_name[:15] + ("..." if len(risk_name) > 15 else ""),  # Truncate long names
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor="black",
                ax=20,
                ay=-20,
                font=dict(size=9, color="black"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="black",
                borderwidth=1
            )

        fig.update_xaxes(range=[0, 1], title="Probability")
        fig.update_yaxes(range=[0, 1], title="Impact")
        fig = apply_pdf_styling(fig)

        # Fix legend positioning for risk matrix to prevent overlap with x-axis
        fig.update_layout(
            margin=dict(l=60, r=60, t=80, b=120),  # Increased bottom margin for legend
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.3,  # Moved legend further down
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="gray",
                borderwidth=1
            )
        )

        return chart_to_base64(fig)
    except Exception as e:
        logger.error(f"Failed to create risk matrix chart: {e}")
        return create_error_chart(f"Risk matrix chart error: {str(e)}")

def create_gantt_chart(data: Dict) -> str:
    """Create Gantt chart for project timeline"""
    try:
        if not validate_chart_data(data, ['tasks', 'start_dates', 'durations']):
            return create_error_chart("Invalid Gantt data: missing tasks, start_dates, or durations")

        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame({
            'Task': data['tasks'],
            'Start': pd.to_datetime(data['start_dates']),
            'Duration': data['durations']
        })

        # Calculate end dates
        df['End'] = df['Start'] + pd.to_timedelta(df['Duration'], unit='D')

        fig = px.timeline(
            df,
            x_start='Start',
            x_end='End',
            y='Task',
            title='Project Timeline (Gantt Chart)',
            color='Task',
            color_discrete_sequence=CHART_CONFIG['color_palette']
        )

        fig.update_yaxes(autorange="reversed")  # Tasks from top to bottom
        fig = apply_pdf_styling(fig)
        return chart_to_base64(fig)
    except Exception as e:
        logger.error(f"Failed to create Gantt chart: {e}")
        return create_error_chart(f"Gantt chart error: {str(e)}")

def validate_chart_data(data: Dict, required_fields: List[str]) -> bool:
    """Validate that chart data contains required fields"""
    if not isinstance(data, dict):
        return False

    for field in required_fields:
        if field not in data or not data[field]:
            logger.warning(f"Chart data missing required field: {field}")
            return False

    return True

def create_chart_section(title: str, chart_html: str, description: str = "") -> str:
    """Create HTML section with chart and description"""
    return f"""
    <div class="chart-section" style="page-break-inside: avoid; margin: 30px 0; padding: 20px; border: 1px solid #e0e0e0; border-radius: 8px;">
        <h3 style="color: #2c3e50; font-size: 18px; margin-bottom: 15px; font-weight: 600;">{title}</h3>
        {f'<p style="color: #5a6c7d; font-size: 14px; margin-bottom: 20px; line-height: 1.5;">{description}</p>' if description else ''}
        <div class="chart-container" style="text-align: center; background: #fafafa; padding: 15px; border-radius: 4px;">
            {chart_html}
        </div>
    </div>
    """

def extract_data_from_proposal(proposal_data: Dict, data_type: str) -> Dict:
    """Extract chart data from actual proposal data"""
    try:
        if data_type == 'budget':
            return extract_budget_data_from_calculation(proposal_data)
        elif data_type == 'timeline':
            return extract_timeline_data_from_generation(proposal_data)
        elif data_type == 'resources':
            return extract_resource_data_from_calculation(proposal_data)
        elif data_type == 'roi':
            return extract_roi_data_from_financial(proposal_data)
        elif data_type == 'risks':
            return extract_risk_data_from_analysis(proposal_data)
        else:
            logger.warning(f"Unknown data type for extraction: {data_type}")
            return {}
    except Exception as e:
        logger.error(f"Failed to extract {data_type} data from proposal: {e}")
        return {}

def extract_budget_data_from_calculation(proposal_data: Dict) -> Dict:
    """Extract budget data from actual cost calculations"""
    try:
        # Look for budget data in proposal sections
        budget_info = proposal_data.get('budget_calculation', {})
        if not budget_info:
            # Try to extract from sections
            for section_name, section_data in proposal_data.items():
                if 'budget' in section_name.lower() or 'cost' in section_name.lower():
                    content = section_data.get('content', '')
                    if 'total_cost' in str(section_data) or '$' in content:
                        budget_info = section_data
                        break

        if isinstance(budget_info, dict) and 'breakdown_by_role' in budget_info:
            # Use actual breakdown data
            breakdown = budget_info['breakdown_by_role']
            return {
                'categories': list(breakdown.keys()),
                'values': list(breakdown.values())
            }
        elif isinstance(budget_info, dict) and 'breakdown' in budget_info:
            # Alternative breakdown format
            breakdown = budget_info['breakdown']
            return {
                'categories': list(breakdown.keys()),
                'values': list(breakdown.values())
            }
        else:
            # Fallback: parse content for cost information
            return parse_budget_from_content(proposal_data)

    except Exception as e:
        logger.error(f"Failed to extract budget data: {e}")
        return get_default_budget_data()

def extract_timeline_data_from_generation(proposal_data: Dict) -> Dict:
    """Extract timeline data from actual timeline generation"""
    try:
        # Look for timeline data in proposal sections
        timeline_info = proposal_data.get('timeline_generation', {})
        if not timeline_info:
            # Try to extract from sections
            for section_name, section_data in proposal_data.items():
                if 'timeline' in section_name.lower() or 'schedule' in section_name.lower():
                    timeline_info = section_data
                    break

        if isinstance(timeline_info, dict) and 'timeline' in timeline_info:
            timeline_phases = timeline_info['timeline']

            # Extract phase information
            phases = []
            costs = []
            cumulative_cost = 0

            for phase in timeline_phases:
                phases.append(phase.get('phase', 'Phase'))
                # Estimate cost per phase (if not available, distribute total evenly)
                phase_cost = phase.get('cost', 0)
                if phase_cost == 0:
                    # Estimate based on duration
                    duration = phase.get('duration_weeks', 4)
                    phase_cost = duration * 25000  # Rough estimate

                costs.append(phase_cost)
                cumulative_cost += phase_cost

            return {
                'months': phases,
                'costs': costs,
                'cumulative': [sum(costs[:i+1]) for i in range(len(costs))]
            }
        else:
            return parse_timeline_from_content(proposal_data)

    except Exception as e:
        logger.error(f"Failed to extract timeline data: {e}")
        return get_default_timeline_data()

def extract_resource_data_from_calculation(proposal_data: Dict) -> Dict:
    """Extract resource allocation data from actual cost calculations"""
    try:
        # Look for resource details in cost calculations
        budget_info = proposal_data.get('budget_calculation', {})
        resource_details = budget_info.get('resource_details', [])

        if resource_details:
            # Group by role and seniority
            roles = {}
            for resource in resource_details:
                role = resource.get('role', 'Developer')
                level = resource.get('level', 'Mid-level').lower()

                if role not in roles:
                    roles[role] = {'senior': 0, 'mid': 0, 'junior': 0}

                if 'senior' in level:
                    roles[role]['senior'] += 1
                elif 'junior' in level:
                    roles[role]['junior'] += 1
                else:
                    roles[role]['mid'] += 1

            return {
                'roles': list(roles.keys()),
                'senior': [roles[role]['senior'] for role in roles.keys()],
                'mid': [roles[role]['mid'] for role in roles.keys()],
                'junior': [roles[role]['junior'] for role in roles.keys()]
            }
        else:
            return parse_resource_from_content(proposal_data)

    except Exception as e:
        logger.error(f"Failed to extract resource data: {e}")
        return get_default_resource_data()

def extract_roi_data_from_financial(proposal_data: Dict) -> Dict:
    """Extract ROI data from financial analysis"""
    try:
        # Look for financial information
        budget_info = proposal_data.get('budget_calculation', {})
        total_cost = budget_info.get('total_cost', 0)
        duration = budget_info.get('duration_months', 6)

        if total_cost > 0:
            # Generate ROI projection based on total cost
            quarters = [f'Q{i+1}' for i in range(8)]

            # Investment curve (front-loaded)
            investment_per_quarter = total_cost / 4
            investment = [investment_per_quarter * min(i+1, 4) for i in range(8)]

            # Returns curve (back-loaded, assuming 150% total return)
            total_return = total_cost * 1.5
            returns = [0, 0, total_return * 0.1, total_return * 0.3,
                      total_return * 0.6, total_return * 0.8,
                      total_return * 1.0, total_return * 1.0]

            # Net value
            net_value = [returns[i] - investment[i] for i in range(8)]

            return {
                'quarters': quarters,
                'investment': investment,
                'returns': returns,
                'net_value': net_value
            }
        else:
            return get_default_roi_data()

    except Exception as e:
        logger.error(f"Failed to extract ROI data: {e}")
        return get_default_roi_data()

def extract_risk_data_from_analysis(proposal_data: Dict) -> Dict:
    """Extract risk data from actual risk analysis"""
    try:
        # Look for risk analysis in proposal sections
        risk_info = proposal_data.get('risk_analysis', [])
        if not risk_info:
            # Try to find risks in sections
            for section_name, section_data in proposal_data.items():
                if 'risk' in section_name.lower():
                    content = section_data.get('content', '')
                    if isinstance(content, list):
                        risk_info = content
                        break

        if risk_info and isinstance(risk_info, list):
            risks = []
            probabilities = []
            impacts = []
            sizes = []

            for risk in risk_info:
                risks.append(risk.get('risk', 'Unknown Risk'))

                # Convert probability strings to numbers
                prob = risk.get('probability', 'Medium')
                prob_map = {'Low': 0.2, 'Medium': 0.5, 'High': 0.8, 'Very High': 0.9}
                probabilities.append(prob_map.get(prob, 0.5))

                # Convert impact strings to numbers
                impact = risk.get('impact', 'Medium')
                impact_map = {'Low': 0.2, 'Medium': 0.5, 'High': 0.8, 'Very High': 0.9}
                impacts.append(impact_map.get(impact, 0.5))

                # Size based on risk score
                risk_score = risk.get('risk_score', 50)
                sizes.append(max(30, min(150, risk_score * 5)))  # Scale to 30-150

            return {
                'risks': risks,
                'probability': probabilities,
                'impact': impacts,
                'sizes': sizes
            }
        else:
            return get_default_risk_data()

    except Exception as e:
        logger.error(f"Failed to extract risk data: {e}")
        return get_default_risk_data()

# Helper functions for content parsing and defaults
def parse_budget_from_content(proposal_data: Dict) -> Dict:
    """Parse budget information from proposal content text"""
    # Default budget structure if no specific data found
    return get_default_budget_data()

def parse_timeline_from_content(proposal_data: Dict) -> Dict:
    """Parse timeline information from proposal content text"""
    return get_default_timeline_data()

def parse_resource_from_content(proposal_data: Dict) -> Dict:
    """Parse resource information from proposal content text"""
    return get_default_resource_data()

# Default data structures (fallbacks)
def get_default_budget_data() -> Dict:
    return {
        'categories': ['Development', 'Infrastructure', 'Management', 'QA', 'Contingency'],
        'values': [300000, 50000, 50000, 50000, 50000]
    }

def get_default_timeline_data() -> Dict:
    return {
        'months': ['Phase 1', 'Phase 2', 'Phase 3', 'Phase 4', 'Phase 5', 'Phase 6'],
        'costs': [80000, 140000, 180000, 160000, 120000, 80000],
        'cumulative': [80000, 220000, 400000, 560000, 680000, 760000]
    }

def get_default_resource_data() -> Dict:
    return {
        'roles': ['Backend', 'Frontend', 'DevOps', 'QA', 'PM'],
        'senior': [2, 1, 1, 0, 1],
        'mid': [3, 2, 1, 2, 0],
        'junior': [2, 3, 1, 2, 0]
    }

def get_default_roi_data() -> Dict:
    return {
        'quarters': ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6'],
        'investment': [200000, 400000, 600000, 760000, 760000, 760000],
        'returns': [0, 50000, 150000, 300000, 500000, 750000],
        'net_value': [-200000, -350000, -450000, -460000, -260000, -10000]
    }

def get_default_risk_data() -> Dict:
    return {
        'risks': ['Technical', 'Budget', 'Timeline', 'Resource', 'Market'],
        'probability': [0.3, 0.2, 0.4, 0.25, 0.15],
        'impact': [0.8, 0.9, 0.7, 0.6, 0.5],
        'sizes': [100, 120, 80, 60, 40]
    }

def select_chart_type(data_type: str, data_structure: Dict) -> str:
    """Automatically select appropriate chart type based on data"""
    chart_mapping = {
        'budget': 'pie',
        'timeline': 'line',
        'resources': 'stacked_bar',
        'roi': 'multi_line',
        'risks': 'scatter',
        'gantt': 'timeline'
    }
    return chart_mapping.get(data_type, 'bar')

def generate_multiple_charts(proposal_content: str, chart_types: List[str]) -> Dict[str, str]:
    """Generate multiple charts from proposal content"""
    charts = {}

    for chart_type in chart_types:
        try:
            data = extract_data_from_proposal(proposal_content, chart_type)

            if chart_type == 'budget':
                charts['budget_breakdown'] = create_budget_pie_chart(data)
            elif chart_type == 'timeline':
                charts['timeline'] = create_timeline_chart(data)
            elif chart_type == 'resources':
                charts['resource_allocation'] = create_resource_chart(data)
            elif chart_type == 'roi':
                charts['roi_projection'] = create_roi_chart(data)
            elif chart_type == 'risks':
                charts['risk_matrix'] = create_risk_matrix_chart(data)
            elif chart_type == 'gantt':
                charts['gantt_chart'] = create_gantt_chart(data)
            else:
                logger.warning(f"Unknown chart type: {chart_type}")

        except Exception as e:
            logger.error(f"Failed to generate {chart_type} chart: {e}")
            charts[f'{chart_type}_error'] = create_error_chart(f"Failed to generate {chart_type} chart")

    return charts
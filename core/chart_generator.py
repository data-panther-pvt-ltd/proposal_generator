"""
Chart Generator Module using Plotly
"""

import json
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any
import os
from pathlib import Path
import logging
import base64
from io import BytesIO

# Chart size limits to prevent context overflow
MAX_CHART_ITEMS = 15  # Maximum number of items in charts
MAX_PHASE_NAME_LENGTH = 50  # Maximum length of phase names
MAX_RISK_NAME_LENGTH = 40  # Maximum length of risk names
MAX_BUDGET_CATEGORY_LENGTH = 40  # Maximum length of budget categories

logger = logging.getLogger(__name__)

class ChartGenerator:
    """Generate charts for proposal sections using Plotly"""
    
    def __init__(self, output_format='static'):
        self.default_colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A4C93']
        self.output_format = output_format  # 'static' for PDF, 'interactive' for HTML
        
    def _fig_to_output(self, fig: go.Figure, div_id: str) -> str:
        """Convert figure to appropriate output format"""
        if self.output_format == 'static':
            # Convert to static image for PDF
            try:
                # Update figure for better PDF appearance
                fig.update_layout(
                    font=dict(size=14, family="Arial, sans-serif"),
                    title_font_size=18,
                    # Respect per-chart legend settings
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    margin=dict(l=60, r=60, t=80, b=60)
                )
                
                # Try to use kaleido for high-quality PNG export
                img_bytes = fig.to_image(
                    format="png", 
                    width=1200,  # Higher resolution
                    height=600, 
                    scale=2  # 2x scale for retina displays
                )
                # Encode as base64
                img_base64 = base64.b64encode(img_bytes).decode()
                # Return as embedded image HTML with better styling
                return f'''
                <div style="text-align: center; margin: 20px 0;">
                    <img src="data:image/png;base64,{img_base64}" 
                         style="width:100%; max-width:900px; height:auto; 
                                border: 1px solid #e0e0e0; border-radius: 8px; 
                                box-shadow: 0 2px 4px rgba(0,0,0,0.1);" />
                </div>
                '''
            except ImportError:
                logger.warning("Kaleido not installed. Install it for better PDF charts: pip install kaleido")
                # Fallback to SVG with inline styles
                try:
                    fig.update_layout(
                        font=dict(size=14),
                        showlegend=True,
                        plot_bgcolor='white',
                        paper_bgcolor='white'
                    )
                    svg = fig.to_html(
                        include_plotlyjs=False,
                        div_id=div_id,
                        config={'staticPlot': True}
                    )
                    # Wrap in a container for better PDF rendering
                    return f'<div style="page-break-inside: avoid; margin: 20px 0;">{svg}</div>'
                except Exception as e:
                    logger.error(f"Failed to create chart: {e}")
                    return self._create_error_chart("Chart generation failed")
        else:
            # Return interactive HTML for web viewing
            return fig.to_html(include_plotlyjs='cdn', div_id=div_id, config={'displayModeBar': False})
        
    def generate_gantt_chart(self, timeline_data: Dict) -> str:
        """Generate a timeline table instead of a Gantt chart."""
        try:
            phases = timeline_data.get('phases', [])

            if not phases:
                # Suppress placeholder when no timeline data is present
                return ''

            # Limit and sanitize
            phases = phases[:MAX_CHART_ITEMS]
            sanitized = []
            for i, phase in enumerate(phases):
                name = str(phase.get('name', f'Phase {i+1}'))[:MAX_PHASE_NAME_LENGTH]
                duration = phase.get('duration', 1)
                if not isinstance(duration, (int, float)) or duration <= 0:
                    duration = 1
                start = phase.get('start') or phase.get('start_date') or ''
                end = phase.get('end') or phase.get('end_date') or ''
                notes = phase.get('description') or phase.get('notes') or ''
                sanitized.append({
                    'name': name,
                    'duration': duration,
                    'start': start,
                    'end': end,
                    'notes': notes
                })

            # Determine optional columns
            any_start = any(p['start'] for p in sanitized)
            any_end = any(p['end'] for p in sanitized)
            any_notes = any(p['notes'] for p in sanitized)

            # Build HTML table (uses global table styles)
            html_parts = []
            html_parts.append('<table class="budget-table">')
            html_parts.append('<thead><tr>')
            html_parts.append('<th>#</th>')
            html_parts.append('<th>Phase</th>')
            if any_start:
                html_parts.append('<th>Start</th>')
            if any_end:
                html_parts.append('<th>End</th>')
            html_parts.append('<th>Duration (weeks)</th>')
            if any_notes:
                html_parts.append('<th>Notes</th>')
            html_parts.append('</tr></thead>')
            html_parts.append('<tbody>')

            for idx, p in enumerate(sanitized, start=1):
                html_parts.append('<tr>')
                html_parts.append(f'<td>{idx}</td>')
                html_parts.append(f'<td>{p["name"]}</td>')
                if any_start:
                    html_parts.append(f'<td>{p["start"]}</td>')
                if any_end:
                    html_parts.append(f'<td>{p["end"]}</td>')
                html_parts.append(f'<td>{int(p["duration"])}</td>')
                if any_notes:
                    html_parts.append(f'<td>{p["notes"]}</td>')
                html_parts.append('</tr>')

            html_parts.append('</tbody>')
            html_parts.append('</table>')

            # Add optional summary under table
            total_duration = sum(int(p['duration']) for p in sanitized)
            start_date = timeline_data.get('start_date', '')
            end_date = timeline_data.get('end_date', '')
            summary_bits = []
            if start_date:
                summary_bits.append(f'Start: {start_date}')
            if end_date:
                summary_bits.append(f'End: {end_date}')
            if total_duration:
                summary_bits.append(f'Total Duration: {int(total_duration)} weeks')
            if summary_bits:
                html_parts.append(f'<div class="page-info">{" | ".join(summary_bits)}</div>')

            return '\n'.join(html_parts)

        except Exception as e:
            logger.error(f"Error generating timeline table: {str(e)}")
            return ''
    
    def create_budget_chart(self, budget_data: Dict) -> str:
        """Create budget breakdown chart with smart selection (pie vs stacked bar)."""
        try:
            # Extract budget breakdown (prefer RFP-provided breakdowns)
            breakdown = budget_data.get('breakdown_by_role') or budget_data.get('breakdown_by_category') or {}
            
            if not breakdown:
                # Attempt to compute from resources using company skill rates (USD)
                resources = budget_data.get('resources') or budget_data.get('resource_details')
                computed_breakdown = self._estimate_costs_from_skills(resources, budget_data)
                if computed_breakdown:
                    breakdown = computed_breakdown
                    categories = list(breakdown.keys())
                    values = list(breakdown.values())
                else:
                    categories = budget_data.get('categories', [])
                    values = budget_data.get('values', [])
            else:
                categories = list(breakdown.keys())
                # Robust numeric parsing for values that may be strings with commas/currency
                parsed_values = []
                for v in breakdown.values():
                    if isinstance(v, (int, float)):
                        parsed_values.append(float(v))
                    elif isinstance(v, str):
                        cleaned = v.strip().replace(',', '')
                        # Remove currency symbols and anything non-numeric except dot and minus
                        import re
                        cleaned = re.sub(r'[^0-9.\-]', '', cleaned)
                        try:
                            parsed_values.append(float(cleaned))
                        except Exception:
                            parsed_values.append(0.0)
                    else:
                        parsed_values.append(0.0)
                values = parsed_values
            
            # If still no data, honor RFP-only requirement and return placeholder
            if not categories or not values:
                return self._create_error_chart("Budget breakdown not specified in RFP")

            # Limit number of categories to prevent context overflow
            if len(categories) > MAX_CHART_ITEMS:
                # Group smaller items into "Other"
                paired_data = list(zip(categories, values))
                paired_data.sort(key=lambda x: x[1], reverse=True)
                
                top_items = paired_data[:MAX_CHART_ITEMS-1]
                other_items = paired_data[MAX_CHART_ITEMS-1:]
                
                categories = [item[0][:MAX_BUDGET_CATEGORY_LENGTH] for item in top_items]
                values = [item[1] for item in top_items]
                
                if other_items:
                    categories.append("Other")
                    values.append(sum(item[1] for item in other_items))
            else:
                # Truncate category names to reasonable length
                categories = [cat[:MAX_BUDGET_CATEGORY_LENGTH] for cat in categories]
            
            # Choose chart type
            use_pie = len(categories) <= 6
            if use_pie:
                fig = go.Figure(data=[go.Pie(
                    labels=categories,
                    values=values,
                    hole=0.3,
                    marker=dict(
                        colors=self.default_colors[:max(1, len(categories))],
                        line=dict(color='white', width=2)
                    ),
                    textinfo='label+percent',
                    textposition='outside',
                    hovertemplate='<b>%{label}</b><br>Amount: $%{value:,.0f}<br>Percentage: %{percent}<extra></extra>'
                )])
            else:
                # Stacked horizontal bar for better readability
                fig = go.Figure()
                cumulative = 0
                for i, (cat, val) in enumerate(zip(categories, values)):
                    fig.add_trace(go.Bar(
                        y=['Budget'],
                        x=[val],
                        name=cat,
                        orientation='h',
                        marker=dict(color=self.default_colors[i % len(self.default_colors)]),
                        hovertemplate=f'<b>{cat}</b><br>Amount: $%{{x:,.0f}}<extra></extra>'
                    ))
            
            # Compute total from values (auto-generated from RFP data)
            computed_total = sum(v for v in values if isinstance(v, (int, float)))

            # Update layout for professional appearance
            fig.update_layout(
                title=dict(
                    text='Project Budget Breakdown by Category (USD)',
                    font=dict(size=20, color='#2E86AB', family='Arial, sans-serif')
                ),
                height=500,
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.05,
                    font=dict(size=12)
                ),
                xaxis=dict(title='Amount (USD)') if not use_pie else None,
                yaxis=dict(title='') if not use_pie else None,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(size=12, family='Arial, sans-serif'),
                margin=dict(l=60 if not use_pie else 50, r=220, t=80, b=60 if not use_pie else 50),
                barmode='stack' if not use_pie else None
            )
            
            # Removed center total annotation per requirement to keep chart clean
            
            return self._fig_to_output(fig, "budget-chart")
            
        except Exception as e:
            logger.error(f"Error generating budget chart: {str(e)}")
            return self._create_error_chart("Failed to generate budget chart")
    
    def build_risk_matrix(self, risks: List[Dict]) -> str:
        """Build risk assessment matrix chart"""
        try:
            if not risks:
                return self._create_error_chart("Risks not specified in RFP")
            
            # Limit number of risks to prevent context overflow
            risks = risks[:MAX_CHART_ITEMS]
            
            # Prepare data for scatter plot
            x_vals = []  # Probability
            y_vals = []  # Impact
            labels = []
            colors = []
            sizes = []
            
            for risk in risks:
                prob = risk.get('probability', 2)
                impact = risk.get('impact', 2)
                name = risk.get('name', 'Unknown Risk')
                
                # Truncate risk names to reasonable length
                name = name[:MAX_RISK_NAME_LENGTH] + ('...' if len(name) > MAX_RISK_NAME_LENGTH else '')
                
                # Normalize probability and impact to 1-5 scale if needed
                if isinstance(prob, str):
                    prob_map = {'low': 1, 'medium': 3, 'high': 5}
                    prob = prob_map.get(prob.lower(), 2)
                if isinstance(impact, str):
                    impact_map = {'low': 1, 'medium': 3, 'high': 5}
                    impact = impact_map.get(impact.lower(), 2)
                    
                # Convert fractional values (0-1) to 1-5 scale
                try:
                    prob_f = float(prob)
                    impact_f = float(impact)
                    if prob_f <= 1.0:
                        prob_f = max(0.2, prob_f) * 5
                    if impact_f <= 1.0:
                        impact_f = max(0.2, impact_f) * 5
                    prob = prob_f
                    impact = impact_f
                except Exception:
                    pass

                # Ensure values are within valid range (1-5)
                prob = max(1.0, min(5.0, float(prob)))
                impact = max(1.0, min(5.0, float(impact)))
                
                x_vals.append(prob)
                y_vals.append(impact)
                labels.append(name)
                
                # Color based on risk level (prob * impact)
                risk_score = prob * impact
                if risk_score <= 4:
                    colors.append('green')
                elif risk_score <= 9:
                    colors.append('orange')
                else:
                    colors.append('red')
                
                sizes.append(20 + risk_score * 3)
            
            # Create scatter plot
            fig = go.Figure()
            
            # Add background zones
            fig.add_shape(type="rect", x0=0, y0=0, x1=2, y1=2,
                         fillcolor="lightgreen", opacity=0.2, layer="below")
            fig.add_shape(type="rect", x0=2, y0=0, x1=5, y1=2,
                         fillcolor="yellow", opacity=0.2, layer="below")
            fig.add_shape(type="rect", x0=0, y0=2, x1=2, y1=5,
                         fillcolor="yellow", opacity=0.2, layer="below")
            fig.add_shape(type="rect", x0=2, y0=2, x1=5, y1=5,
                         fillcolor="lightcoral", opacity=0.2, layer="below")
            
            # Add risk points
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='markers+text',
                text=labels,
                textposition='top center',
                marker=dict(
                    size=sizes,
                    color=colors,
                    line=dict(color='white', width=2),
                    opacity=0.8
                ),
                hovertemplate='<b>%{text}</b><br>Probability: %{x:.1f}<br>Impact: %{y:.1f}<extra></extra>',
                showlegend=False
            ))
            
            # Update layout
            fig.update_layout(
                title='Risk Matrix: Probability vs Impact',
                xaxis=dict(title='Probability (1-5)', range=[1, 5], tickmode='linear', tick0=1, dtick=1),
                yaxis=dict(title='Impact (1-5)', range=[1, 5], tickmode='linear', tick0=1, dtick=1),
                height=500,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            return self._fig_to_output(fig, "risk-matrix")
            
        except Exception as e:
            logger.error(f"Error generating risk matrix: {str(e)}")
            return self._create_error_chart("Failed to generate risk matrix")
    
    def generate_resource_chart(self, resource_data: Dict) -> str:
        """Generate resource allocation chart"""
        try:
            team = resource_data.get('recommended_team', {})
            
            if not team:
                return self._create_error_chart("No resource data available")
            
            # Create bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=list(team.keys()),
                    y=list(team.values()),
                    text=list(team.values()),
                    textposition='auto',
                    marker_color=self.default_colors[0],
                    hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
                )
            ])
            
            # Update layout
            fig.update_layout(
                title='Team Composition',
                xaxis_title='Role',
                yaxis_title='Number of Resources',
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            return self._fig_to_output(fig, "resource-chart")
            
        except Exception as e:
            logger.error(f"Error generating resource chart: {str(e)}")
            return self._create_error_chart("Failed to generate resource chart")
    
    def _get_default_risks(self) -> List[Dict]:
        """Get default risks for demonstration"""
        return [
            {"name": "Scope Creep", "probability": 3, "impact": 4},
            {"name": "Resource Availability", "probability": 2, "impact": 3},
            {"name": "Technical Complexity", "probability": 3, "impact": 3},
            {"name": "Budget Overrun", "probability": 2, "impact": 4},
            {"name": "Timeline Delays", "probability": 3, "impact": 3}
        ]
    
    def generate_bar_chart(self, data: Dict) -> str:
        """Generate a generic bar chart for any section"""
        try:
            # Extract data from the input
            title = data.get('title', 'Data Comparison')
            categories = data.get('categories', [])
            values = data.get('values', [])
            x_label = data.get('x_label', 'Categories')
            y_label = data.get('y_label', 'Values')
            
            # If no data provided, create sample data
            if not categories or not values:
                return self._fig_to_output(
                    go.Figure().add_annotation(text='No RFP-aligned data available for this chart', x=0.5, y=0.5, showarrow=False),
                    f"bar-chart-empty"
                )
            
            # Limit items to prevent overflow
            categories = categories[:MAX_CHART_ITEMS]
            values = values[:MAX_CHART_ITEMS]
            
            # Create bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=categories,
                    y=values,
                    text=values,
                    textposition='auto',
                    marker_color=self.default_colors[0],
                    hovertemplate='<b>%{x}</b><br>Value: %{y}<extra></extra>'
                )
            ])
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title=x_label,
                yaxis_title=y_label,
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12),
                margin=dict(l=20, r=20, t=40, b=20),
                showlegend=False
            )
            
            return self._fig_to_output(fig, f"bar-chart-{title.replace(' ', '-').lower()}")
            
        except Exception as e:
            logger.error(f"Error generating bar chart: {str(e)}")
            return self._create_error_chart("Failed to generate bar chart")
    
    def generate_pie_chart(self, data: Dict) -> str:
        """Generate a generic pie chart for any section"""
        try:
            # Extract data from the input
            title = data.get('title', 'Distribution Analysis')
            labels = data.get('labels', [])
            values = data.get('values', [])
            
            # If no data provided, create sample data
            if not labels or not values:
                labels = ['Research', 'Development', 'Testing', 'Deployment', 'Support']
                values = [15, 40, 20, 15, 10]
                title = title or 'Resource Allocation'
            
            # Limit items to prevent overflow
            labels = labels[:MAX_CHART_ITEMS]
            values = values[:MAX_CHART_ITEMS]
            
            # Create pie chart
            fig = go.Figure(data=[
                go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.3,  # Create a donut chart for modern look
                    marker=dict(colors=self.default_colors[:len(labels)]),
                    textinfo='label+percent',
                    hovertemplate='<b>%{label}</b><br>Value: %{value}<br>Percentage: %{percent}<extra></extra>'
                )
            ])
            
            # Update layout
            fig.update_layout(
                title=title,
                height=450,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12),
                margin=dict(l=20, r=20, t=40, b=20),
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.05
                )
            )
            
            return self._fig_to_output(fig, f"pie-chart-{title.replace(' ', '-').lower()}")
            
        except Exception as e:
            logger.error(f"Error generating pie chart: {str(e)}")
            return self._create_error_chart("Failed to generate pie chart")
    
    def generate_line_chart(self, data: Dict) -> str:
        """Generate a generic line chart for any section"""
        try:
            # Extract data from the input
            title = data.get('title', 'Trend Analysis')
            x_data = data.get('x_data', [])
            y_data = data.get('y_data', [])
            x_label = data.get('x_label', 'Timeline')
            y_label = data.get('y_label', 'Progress')
            
            # If no data provided, create sample data
            if not x_data or not y_data:
                x_data = ['Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5', 'Week 6', 'Week 7', 'Week 8']
                y_data = [10, 25, 40, 55, 65, 75, 85, 95]
                title = title or 'Project Progress Over Time'
            
            # Limit items to prevent overflow
            x_data = x_data[:MAX_CHART_ITEMS]
            y_data = y_data[:MAX_CHART_ITEMS]
            
            # Create line chart
            fig = go.Figure(data=[
                go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode='lines+markers',
                    name='Progress',
                    line=dict(color=self.default_colors[0], width=3),
                    marker=dict(size=8),
                    hovertemplate='<b>%{x}</b><br>Value: %{y}<extra></extra>'
                )
            ])
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title=x_label,
                yaxis_title=y_label,
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12),
                margin=dict(l=20, r=20, t=40, b=20),
                showlegend=False,
                xaxis=dict(showgrid=True, gridcolor='rgba(200,200,200,0.3)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(200,200,200,0.3)')
            )
            
            return self._fig_to_output(fig, f"line-chart-{title.replace(' ', '-').lower()}")
            
        except Exception as e:
            logger.error(f"Error generating line chart: {str(e)}")
            return self._create_error_chart("Failed to generate line chart")
    
    def _create_error_chart(self, message: str) -> str:
        """Create a placeholder chart for errors"""
        return f'<div style="padding: 20px; text-align: center; background: #f8f9fa; border-radius: 8px; color: #666;">{message}</div>'

    def _estimate_costs_from_skills(self, resources: Any, budget_data: Dict) -> Dict[str, float]:
        """Compute role-based USD costs using skill_company.csv and skill_external.csv.

        resources: list of items with fields role, level, hours, source (internal/external)
        Returns breakdown dict {role: cost_usd}
        """
        try:
            if not resources or not isinstance(resources, (list, tuple)):
                return {}

            # Load skill CSVs relative to project root
            data_dir = Path(__file__).resolve().parent.parent / 'data'
            internal_path = data_dir / 'skill_company.csv'
            external_path = data_dir / 'skill_external.csv'
            import pandas as pd
            internal_df = pd.read_csv(internal_path) if internal_path.exists() else None
            external_df = pd.read_csv(external_path) if external_path.exists() else None

            breakdown: Dict[str, float] = {}

            def find_rate(role: str, level: str, source: str) -> float:
                df = internal_df if source == 'internal' else external_df
                if df is None or df.empty:
                    # conservative defaults in USD
                    defaults = {'Junior': 50, 'Mid-level': 80, 'Senior': 120}
                    if source != 'internal':
                        defaults = {'Junior': 70, 'Mid-level': 110, 'Senior': 160}
                    return float(defaults.get(level, 80 if source == 'internal' else 110))

                # Try match by skill_name or skill_category
                mask = (
                    df['skill_name'].astype(str).str.contains(role, case=False, na=False) |
                    df['skill_category'].astype(str).str.contains(role, case=False, na=False)
                )
                candidates = df[mask]
                if candidates.empty:
                    return float(80 if source == 'internal' else 110)
                # Prefer level match
                level_mask = candidates['experience_level'].astype(str).str.contains(level, case=False, na=False)
                if candidates[level_mask].empty:
                    return float(candidates.iloc[0]['hourly_rate_usd'])
                return float(candidates[level_mask].iloc[0]['hourly_rate_usd'])

            for res in resources:
                role = str(res.get('role', 'Resource'))
                level = str(res.get('level', 'Mid-level'))
                hours = res.get('hours', 160)
                try:
                    hours = float(hours)
                except Exception:
                    hours = 160.0
                source = str(res.get('source', 'internal')).lower()
                if source not in ('internal', 'external'):
                    source = 'internal'
                rate = find_rate(role, level, source)
                cost = rate * hours
                breakdown[role] = breakdown.get(role, 0.0) + float(cost)

            # Ensure USD by nature of using *_rate_usd columns
            return {k: float(v) for k, v in breakdown.items()}
        except Exception as e:
            logger.error(f"Skill-based cost estimation failed: {e}")
            return {}
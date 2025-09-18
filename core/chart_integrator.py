"""
Chart Integration Service for Proposal Generation
Handles automatic chart generation and embedding based on proposal data
"""

import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import pandas as pd

from tools.proposal_tools import generate_proposal_charts
from utils.data_loader import DataLoader

# Import chart generation tools
try:
    from tools.chart_tools import (
        create_budget_pie_chart,
        create_timeline_chart,
        create_resource_chart,
        create_roi_chart,
        create_risk_matrix_chart,
        create_chart_section,
        extract_data_from_proposal
    )
    CHARTS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Chart tools not available: {e}")
    CHARTS_AVAILABLE = False

logger = logging.getLogger(__name__)

class ChartIntegrator:
    """Service for integrating charts into proposals"""

    def __init__(self, config: Dict):
        self.config = config
        self.chart_config = config.get('charts', {})
        self.enabled = self.chart_config.get('enabled', True) and CHARTS_AVAILABLE
        self.auto_generate = self.chart_config.get('auto_generate', True)

        # Chart section mappings from config
        self.chart_sections = self.chart_config.get('chart_sections', {})
        self.default_chart_types = self.chart_config.get('chart_types', ['budget', 'timeline'])

        logger.info(f"ChartIntegrator initialized - enabled: {self.enabled}, auto_generate: {self.auto_generate}")

    def should_generate_charts_for_section(self, section_name: str) -> bool:
        """Check if charts should be generated for a specific section"""
        if not self.enabled or not self.auto_generate:
            return False

        return section_name in self.chart_sections

    def get_chart_types_for_section(self, section_name: str) -> List[str]:
        """Get chart types that should be generated for a section"""
        if section_name in self.chart_sections:
            return self.chart_sections[section_name].get('charts', [])
        return []

    def generate_charts_for_proposal(self, proposal_data: Dict) -> Dict[str, Any]:
        """Generate all charts for the proposal based on available data"""
        if not self.enabled:
            logger.info("Chart generation disabled")
            return {}

        try:
            # Collect all data needed for chart generation
            chart_data = self._extract_chart_data_from_proposal(proposal_data)

            if not chart_data:
                logger.warning("No data available for chart generation")
                return {}

            # Generate charts using the chart generation tool
            charts_json = generate_proposal_charts(
                proposal_data=json.dumps(chart_data),
                chart_types=json.dumps(self.default_chart_types)
            )

            charts_result = json.loads(charts_json)

            if 'error' in charts_result:
                logger.error(f"Chart generation error: {charts_result['error']}")
                return {}

            logger.info(f"Generated {charts_result.get('metadata', {}).get('charts_generated', 0)} charts")
            return charts_result

        except Exception as e:
            logger.error(f"Failed to generate charts: {e}")
            return {}

    def generate_charts_for_section(self, section_name: str, section_data: Dict, proposal_data: Dict) -> Optional[str]:
        """Generate charts for a specific section"""
        if not self.should_generate_charts_for_section(section_name):
            return None

        try:
            # Get chart types for this section
            chart_types = self.get_chart_types_for_section(section_name)
            if not chart_types:
                return None

            charts_html = []

            for chart_type in chart_types:
                try:
                    # Extract specific data for this chart type
                    chart_data = extract_data_from_proposal(proposal_data, chart_type)

                    if not chart_data:
                        logger.warning(f"No data available for {chart_type} chart")
                        continue

                    chart_html = None
                    title = ""
                    description = ""

                    if chart_type == "budget":
                        chart_html = create_budget_pie_chart(chart_data)
                        title = "Budget Breakdown"
                        description = "Visual breakdown of project costs by category"
                    elif chart_type == "timeline":
                        chart_html = create_timeline_chart(chart_data)
                        title = "Project Timeline"
                        description = "Cost distribution and cumulative spend over project duration"
                    elif chart_type == "resources":
                        chart_html = create_resource_chart(chart_data)
                        title = "Resource Allocation"
                        description = "Team composition showing seniority levels by role"
                    elif chart_type == "roi":
                        chart_html = create_roi_chart(chart_data)
                        title = "ROI Projection"
                        description = "Return on investment analysis over time"
                    elif chart_type == "risks":
                        chart_html = create_risk_matrix_chart(chart_data)
                        title = "Risk Assessment Matrix"
                        description = "Risk analysis showing probability vs impact of identified risks"

                    if chart_html:
                        section_html = create_chart_section(title, chart_html, description)
                        charts_html.append(section_html)
                        logger.info(f"Generated {chart_type} chart for section {section_name}")

                except Exception as chart_error:
                    logger.error(f"Failed to generate {chart_type} chart: {chart_error}")
                    continue

            if charts_html:
                return '\n'.join(charts_html)

            return None

        except Exception as e:
            logger.error(f"Failed to generate charts for section {section_name}: {e}")
            return None

    def embed_charts_in_sections(self, proposal_data: Dict) -> Dict:
        """Embed charts directly into section content"""
        if not self.enabled:
            return proposal_data

        modified_proposal = proposal_data.copy()

        # Get sections
        sections = modified_proposal.get('generated_sections', {})
        if not sections:
            sections = modified_proposal.get('sections', {})

        # Add fallback sample data for testing if no real data exists
        if not self._has_chart_data(proposal_data):
            logger.warning("No chart data found - using sample data for demonstration")
            proposal_data = self._add_sample_chart_data(proposal_data)

        for section_name, section_data in sections.items():
            # Generate charts for this section
            charts_html = self.generate_charts_for_section(section_name, section_data, proposal_data)

            if charts_html:
                # Embed charts in section content
                position = self.chart_sections.get(section_name, {}).get('position', 'after_content')

                if isinstance(section_data, dict):
                    current_content = section_data.get('content', '')
                elif isinstance(section_data, str):
                    current_content = section_data
                else:
                    current_content = str(section_data)

                if position == 'after_content':
                    enhanced_content = current_content + '\n\n' + charts_html
                elif position == 'before_content':
                    enhanced_content = charts_html + '\n\n' + current_content
                else:  # replace_content
                    enhanced_content = charts_html

                # Update section with enhanced content
                if isinstance(section_data, dict):
                    sections[section_name]['content'] = enhanced_content
                    sections[section_name]['has_charts'] = True
                else:
                    sections[section_name] = enhanced_content

                logger.info(f"Added charts to section: {section_name}")

        # Update the proposal data
        if 'generated_sections' in modified_proposal:
            modified_proposal['generated_sections'] = sections
        else:
            modified_proposal['sections'] = sections

        return modified_proposal

    def _has_chart_data(self, proposal_data: Dict) -> bool:
        """Check if proposal has any chart-worthy data"""
        return (
            'budget_calculation' in proposal_data or
            'timeline_generation' in proposal_data or
            any(
                isinstance(section, dict) and ('budget_calculation' in section or 'timeline_generation' in section)
                for section in proposal_data.get('generated_sections', {}).values()
            )
        )

    def _add_sample_chart_data(self, proposal_data: Dict) -> Dict:
        """Add realistic chart data based on proposal context and timeline"""
        enhanced_data = proposal_data.copy()

        # Extract project context for realistic data generation
        timeline = self._extract_timeline_duration(proposal_data)
        project_type = self._extract_project_type(proposal_data)

        # Generate realistic budget based on project type and timeline
        enhanced_data['budget_calculation'] = self._generate_realistic_budget(project_type, timeline)

        # Generate realistic timeline data
        enhanced_data['timeline_generation'] = self._generate_realistic_timeline(timeline)

        # Generate realistic ROI projection
        total_cost = enhanced_data['budget_calculation']['total_cost']
        enhanced_data['roi_projection'] = self._generate_realistic_roi(total_cost, timeline)

        # Generate realistic risk data
        enhanced_data['risk_analysis'] = self._generate_realistic_risks(project_type)

        return enhanced_data

    def _extract_timeline_duration(self, proposal_data: Dict) -> int:
        """Extract project timeline in months from proposal data"""
        # Try to extract from request
        request_str = str(proposal_data.get('request', ''))
        if 'timeline=' in request_str:
            timeline_part = request_str.split("timeline='")[1].split("'")[0]
            if 'month' in timeline_part.lower():
                try:
                    return int(timeline_part.split()[0])
                except:
                    pass

        # Default to 12 months for enterprise projects
        return 12

    def _extract_project_type(self, proposal_data: Dict) -> str:
        """Extract project type from proposal data"""
        request_str = str(proposal_data.get('request', ''))
        if 'ai_solution' in request_str.lower():
            return 'ai_platform'
        elif 'web' in request_str.lower():
            return 'web_development'
        elif 'mobile' in request_str.lower():
            return 'mobile_app'
        else:
            return 'software_platform'

    def _generate_realistic_budget(self, project_type: str, timeline_months: int) -> Dict:
        """Generate realistic budget based on project type and timeline"""
        # Load role costs from CSV data
        role_costs = self._load_role_costs_from_csv(project_type)

        # Scale by timeline
        final_costs = {}
        for role, monthly_cost in role_costs.items():
            final_costs[role] = monthly_cost * timeline_months

        total_cost = sum(final_costs.values())

        return {
            'breakdown_by_role': final_costs,
            'total_cost': total_cost,
            'resource_details': [
                {'role': role, 'level': 'Mixed', 'hours': timeline_months * 160}
                for role in final_costs.keys()
            ]
        }

    def _load_role_costs_from_csv(self, project_type: str) -> Dict[str, float]:
        """Load role costs from skill_company.csv based on project type"""
        try:
            # Initialize data loader
            data_loader = DataLoader(self.config)
            skills_df = data_loader.load_skills_data()

            if skills_df.empty:
                logger.warning("No skills data available, falling back to default costs")
                return self._get_fallback_role_costs(project_type)

            # Group by skill category and experience level to calculate average costs
            role_costs = {}

            # Map skill categories to role names based on project type
            role_mapping = self._get_role_mapping(project_type)

            for role_name, skill_categories in role_mapping.items():
                total_cost = 0
                total_employees = 0

                for category in skill_categories:
                    # Filter by skill category
                    category_data = skills_df[skills_df['skill_category'].str.contains(category, case=False, na=False)]

                    if not category_data.empty:
                        # Calculate weighted average based on employee count and hourly rate
                        # Assuming 160 hours per month (40 hours/week * 4 weeks)
                        for _, row in category_data.iterrows():
                            monthly_rate = row['hourly_rate_usd'] * 160
                            employee_count = row['employee_count']
                            total_cost += monthly_rate * employee_count
                            total_employees += employee_count

                if total_employees > 0:
                    role_costs[role_name] = total_cost / total_employees
                else:
                    # Fallback to default if no data found for this role
                    logger.warning(f"No data found for role {role_name}, using fallback cost")
                    fallback_costs = self._get_fallback_role_costs(project_type)
                    role_costs[role_name] = fallback_costs.get(role_name, 75000)

            return role_costs

        except Exception as e:
            logger.error(f"Error loading role costs from CSV: {e}")
            return self._get_fallback_role_costs(project_type)

    def _get_role_mapping(self, project_type: str) -> Dict[str, List[str]]:
        """Get mapping of role names to skill categories based on project type"""
        base_mapping = {
            'Backend Developers': ['Backend Development'],
            'Frontend Developers': ['Frontend Development'],
            'DevOps Engineers': ['DevOps'],
            'QA Engineers': ['Quality Assurance'],
            'Project Managers': ['Project Management'],
            'Full Stack Developers': ['Full Stack Development'],
        }

        # Add project-specific roles
        if project_type == 'ai_platform':
            base_mapping.update({
                'AI/ML Engineers': ['Data Science'],
                'Data Engineers': ['Data Science'],
                'Security Engineers': ['Security'],
            })
        elif project_type == 'mobile_app':
            base_mapping.update({
                'Mobile Developers': ['Mobile Development'],
                'UI/UX Designers': ['Frontend Development'],  # Map to closest available category
            })

        return base_mapping

    def _get_fallback_role_costs(self, project_type: str) -> Dict[str, float]:
        """Fallback role costs if CSV data is not available"""
        role_costs = {
            'Backend Developers': 85000,   # Mix of senior/mid level
            'Frontend Developers': 75000,  # Mix of senior/mid level
            'DevOps Engineers': 90000,     # Senior level
            'QA Engineers': 55000,         # Mix of levels
            'Project Managers': 80000,     # Senior level
        }

        # Adjust for project type
        if project_type == 'ai_platform':
            role_costs['AI/ML Engineers'] = 110000
            role_costs['Data Engineers'] = 75000
            role_costs['Security Engineers'] = 95000
        elif project_type == 'mobile_app':
            role_costs['Mobile Developers'] = 80000
            role_costs['UI/UX Designers'] = 60000

        return role_costs

    def _generate_realistic_timeline(self, timeline_months: int) -> Dict:
        """Generate realistic timeline phases based on project duration"""
        if timeline_months <= 6:
            phases = [
                {'phase': 'Discovery & Planning', 'duration_weeks': 4, 'percentage': 0.2},
                {'phase': 'Development', 'duration_weeks': timeline_months * 3, 'percentage': 0.6},
                {'phase': 'Testing & Deployment', 'duration_weeks': 4, 'percentage': 0.2}
            ]
        else:
            phases = [
                {'phase': 'Discovery & Requirements', 'duration_weeks': 6, 'percentage': 0.15},
                {'phase': 'Architecture & Design', 'duration_weeks': 6, 'percentage': 0.15},
                {'phase': 'Development Phase 1', 'duration_weeks': timeline_months * 2, 'percentage': 0.35},
                {'phase': 'Development Phase 2', 'duration_weeks': timeline_months * 1.5, 'percentage': 0.25},
                {'phase': 'Testing & QA', 'duration_weeks': 4, 'percentage': 0.08},
                {'phase': 'Deployment & Training', 'duration_weeks': 2, 'percentage': 0.02}
            ]

        return {'timeline': phases}

    def _generate_realistic_roi(self, total_cost: int, timeline_months: int) -> Dict:
        """Generate realistic ROI projection based on total cost"""
        quarters = timeline_months // 3 + 2  # Project duration + 2 quarters
        quarter_labels = [f'Q{i+1}' for i in range(quarters)]

        # Investment curve (front-loaded during development)
        investment_per_quarter = total_cost / max(timeline_months // 3, 1)
        investment = []
        for i in range(quarters):
            if i < timeline_months // 3:
                investment.append(investment_per_quarter * (i + 1))
            else:
                investment.append(total_cost)

        # Returns curve (back-loaded after deployment)
        annual_savings = total_cost * 0.4  # 40% annual savings assumption
        quarterly_savings = annual_savings / 4
        returns = []
        for i in range(quarters):
            if i < timeline_months // 3:
                returns.append(quarterly_savings * 0.1 * i)  # Minimal returns during development
            else:
                returns.append(quarterly_savings * (i - timeline_months // 3 + 1))

        # Net value calculation
        net_value = [returns[i] - investment[i] for i in range(quarters)]

        return {
            'quarters': quarter_labels,
            'investment': investment,
            'projected_savings': returns,
            'net_value': net_value
        }

    def _generate_realistic_risks(self, project_type: str) -> List[Dict]:
        """Generate realistic risk data based on project type"""
        # Base risks common to all projects
        base_risks = [
            {'risk': 'Budget Overrun', 'probability': 'Medium', 'impact': 'High', 'risk_score': 15},
            {'risk': 'Timeline Delays', 'probability': 'Medium', 'impact': 'Medium', 'risk_score': 12},
            {'risk': 'Resource Availability', 'probability': 'Low', 'impact': 'Medium', 'risk_score': 8},
            {'risk': 'Technical Complexity', 'probability': 'Medium', 'impact': 'Medium', 'risk_score': 12},
            {'risk': 'Client Requirements Change', 'probability': 'High', 'impact': 'Medium', 'risk_score': 16}
        ]

        # Project-specific risks
        if project_type == 'ai_platform':
            base_risks.extend([
                {'risk': 'Data Quality Issues', 'probability': 'Medium', 'impact': 'High', 'risk_score': 15},
                {'risk': 'Algorithm Performance', 'probability': 'Medium', 'impact': 'Medium', 'risk_score': 12},
                {'risk': 'Cybersecurity Compliance', 'probability': 'Low', 'impact': 'Very High', 'risk_score': 18}
            ])
        elif project_type == 'mobile_app':
            base_risks.extend([
                {'risk': 'App Store Approval', 'probability': 'Low', 'impact': 'Medium', 'risk_score': 8},
                {'risk': 'Device Compatibility', 'probability': 'Medium', 'impact': 'Low', 'risk_score': 6},
                {'risk': 'Performance on Older Devices', 'probability': 'Medium', 'impact': 'Medium', 'risk_score': 10}
            ])
        elif project_type == 'web_development':
            base_risks.extend([
                {'risk': 'Browser Compatibility', 'probability': 'Low', 'impact': 'Low', 'risk_score': 4},
                {'risk': 'SEO Requirements', 'probability': 'Medium', 'impact': 'Low', 'risk_score': 6},
                {'risk': 'Third-party Integration', 'probability': 'Medium', 'impact': 'Medium', 'risk_score': 10}
            ])

        return base_risks

    def _extract_chart_data_from_proposal(self, proposal_data: Dict) -> Dict:
        """Extract data from proposal suitable for chart generation"""
        chart_data = {}

        try:
            # Extract budget calculation data
            if 'budget_calculation' in proposal_data:
                chart_data['budget_calculation'] = proposal_data['budget_calculation']

            # Extract timeline generation data
            if 'timeline_generation' in proposal_data:
                chart_data['timeline_generation'] = proposal_data['timeline_generation']

            # Extract risk analysis data
            if 'risk_analysis' in proposal_data:
                chart_data['risk_analysis'] = proposal_data['risk_analysis']

            # Look for data in sections
            sections = proposal_data.get('generated_sections', proposal_data.get('sections', {}))

            # Extract budget data from Budget section
            budget_section = sections.get('Budget', {})
            if isinstance(budget_section, dict) and 'budget_calculation' in budget_section:
                chart_data['budget_calculation'] = budget_section['budget_calculation']

            # Extract timeline data from timeline sections
            timeline_section = sections.get('Project Plan and Timelines', {})
            if isinstance(timeline_section, dict) and 'timeline_generation' in timeline_section:
                chart_data['timeline_generation'] = timeline_section['timeline_generation']

            # Look for any structured data that might be useful for charts
            for section_name, section_content in sections.items():
                if isinstance(section_content, dict):
                    # Check for calculation results
                    if 'calculations' in section_content:
                        chart_data[f'{section_name.lower()}_calculations'] = section_content['calculations']

                    # Check for timeline data
                    if 'timeline' in section_content:
                        chart_data[f'{section_name.lower()}_timeline'] = section_content['timeline']

        except Exception as e:
            logger.error(f"Error extracting chart data: {e}")

        return chart_data

    def add_charts_section(self, proposal_data: Dict) -> Dict:
        """Add a dedicated charts section to the proposal"""
        if not self.enabled:
            return proposal_data

        try:
            # Generate comprehensive charts for the proposal
            charts_result = self.generate_charts_for_proposal(proposal_data)

            if not charts_result or 'charts' not in charts_result:
                return proposal_data

            # Create charts section content
            chart_sections = charts_result.get('chart_sections', {})
            if chart_sections:
                charts_html = f"""
                <div class="charts-overview">
                    <h2>Project Analytics</h2>
                    <p>Visual analysis of key project metrics and projections.</p>
                    {''.join(chart_sections.values())}
                </div>
                """

                # Add to proposal sections
                modified_proposal = proposal_data.copy()
                sections = modified_proposal.get('generated_sections', modified_proposal.get('sections', {}))
                sections['Project Analytics'] = {
                    'content': charts_html,
                    'type': 'charts_overview',
                    'has_charts': True
                }

                if 'generated_sections' in modified_proposal:
                    modified_proposal['generated_sections'] = sections
                else:
                    modified_proposal['sections'] = sections

                logger.info("Added dedicated charts section to proposal")

        except Exception as e:
            logger.error(f"Failed to add charts section: {e}")

        return proposal_data
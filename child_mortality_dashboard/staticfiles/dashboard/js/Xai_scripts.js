// SHAP Values Visualization
function renderSHAPPlot(shapValues) {
    const data = [{
        type: 'bar',
        x: shapValues.map(v => v.value),
        y: shapValues.map(v => v.feature),
        orientation: 'h',
        marker: {
            color: 'rgba(58,71,80,0.6)',
            line: {
                color: 'rgba(58,71,80,1)',
                width: 1.5
            }
        }
    }];

    const layout = {
        title: 'Feature Importance (SHAP Values)',
        xaxis: { title: 'SHAP Value' },
        yaxis: { title: 'Features' },
        height: 300
    };

    Plotly.newPlot('shap-plot', data, layout);
}

// 3D Scatter Plot for Multidimensional Risk Analysis
function render3DScatterPlot(data) {
    const trace = {
        x: data.x,
        y: data.y,
        z: data.z,
        mode: 'markers',
        marker: {
            size: 10,
            color: data.color,
            colorscale: 'Viridis',
            opacity: 0.8
        },
        type: 'scatter3d'
    };

    const layout = {
        title: 'Multidimensional Risk Factors',
        scene: {
            xaxis: { title: data.xLabel },
            yaxis: { title: data.yLabel },
            zaxis: { title: data.zLabel }
        }
    };

    Plotly.newPlot('3d-scatter-plot', [trace], layout);
}

// Sankey Diagram for Risk Factor Interactions
function renderSankeyDiagram(links) {
    const data = [{
        type: "sankey",
        node: {
            pad: 15,
            thickness: 20,
            line: { color: "black", width: 0.5 },
            label: links.labels,
            color: links.colors
        },
        link: {
            source: links.sources,
            target: links.targets,
            value: links.values
        }
    }];

    const layout = {
        title: "Risk Factor Interactions",
        font: { size: 10 }
    };

    Plotly.newPlot('sankey-diagram', data, layout);
}

// Initialize visualizations when data is available
document.addEventListener('DOMContentLoaded', function () {
    // These would typically be populated with actual data from Django context
    const shapValues = JSON.parse('{{ shap_values|safe }}');
    const scatterData = JSON.parse('{{ scatter_plot_data|safe }}');
    const sankeyLinks = JSON.parse('{{ sankey_links|safe }}');

    if (shapValues) renderSHAPPlot(shapValues);
    if (scatterData) render3DScatterPlot(scatterData);
    if (sankeyLinks) renderSankeyDiagram(sankeyLinks);
});
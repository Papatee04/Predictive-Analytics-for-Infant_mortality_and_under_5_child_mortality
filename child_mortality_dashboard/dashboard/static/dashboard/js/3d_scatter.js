document.addEventListener('DOMContentLoaded', function() {
    try {
        const plotData = JSON.parse('{{ 3d_risk_scatter_plot|safe }}');
        
        if (plotData.data && plotData.data.length > 0) {
            Plotly.newPlot('3d-scatter-plot', 
                plotData.data, 
                plotData.layout, 
                {responsive: true}
            );
        } else {
            console.log('No data available for 3D scatter plot');
            document.getElementById('3d-scatter-plot').innerHTML = 'Unable to generate plot';
        }
    } catch (error) {
        console.error('Error parsing plot data:', error);
        document.getElementById('3d-scatter-plot').innerHTML = 'Error loading plot';
    }
});
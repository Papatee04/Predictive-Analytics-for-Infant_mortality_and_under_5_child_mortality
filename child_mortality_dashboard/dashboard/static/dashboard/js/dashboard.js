document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM fully loaded and parsed');
    
    const numberOfChildrenField = document.getElementById('id_number_of_children');
    const maritalStatusField = document.getElementById('id_marital_status');

    if (maritalStatusField && numberOfChildrenField) {
      maritalStatusField.addEventListener('change', function() {
        if (this.value === '0') {
          numberOfChildrenField.disabled = true;
          numberOfChildrenField.value = 0;
        } else {
          numberOfChildrenField.disabled = false;
        }
      });
    }

    // Destructure components from Recharts
    const { 
      BarChart, 
      Bar, 
      XAxis, 
      YAxis, 
      CartesianGrid, 
      Tooltip, 
      Legend, 
      ResponsiveContainer 
    } = Recharts;

    const ModelComparisonDashboard = () => {
      const modelData = [
        { metric: 'Accuracy', 'Logistic Regression': 0.999269, 'Random Forest': 1 },
        { metric: 'Precision', 'Logistic Regression': 0.9990234, 'Random Forest': 1 },
        { metric: 'Recall', 'Logistic Regression': 1, 'Random Forest': 1 },
        { metric: 'AUC', 'Logistic Regression': 0.9985528, 'Random Forest': 1 },
      ];

      const keyInsights = [
        "Maternal health and obstetric care are critical factors in child mortality.",
        "Family size and resource strain impact child survival rates.",
        "Marital status and social support affect child mortality risks.",
        "Maternal education and family planning are associated with lower child mortality.",
      ];

      return (
        <div className="dashboard-container">
          <h2 className="text-xl font-bold mb-4">Model Comparison</h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={modelData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="metric" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="Logistic Regression" fill="#8884d8" />
              <Bar dataKey="Random Forest" fill="#82ca9d" />
            </BarChart>
          </ResponsiveContainer>

          <div className="mt-8">
            <h2 className="text-xl font-bold mb-4">Key Insights</h2>
            <ul className="list-disc pl-5">
              {keyInsights.map((insight, index) => (
                <li key={index} className="mb-2">{insight}</li>
              ))}
            </ul>
          </div>
        </div>
      );
    };

    const rootElement = document.getElementById('dashboard-root');
    if (rootElement) {
        const root = ReactDOM.createRoot(rootElement);
        console.log('Rendering React component');
        root.render(<ModelComparisonDashboard />);
    } else {
        console.error('dashboard-root element not found');
    }
});

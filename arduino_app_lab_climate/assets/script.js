document.addEventListener("DOMContentLoaded", () => {
    const tempVal = document.getElementById("temp-val");
    const humVal = document.getElementById("hum-val");
    const lightVal = document.getElementById("light-val");

    const maxDataPoints = 20;

    // Chart configs
    const commonOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
            x: { display: false },
            y: { 
                beginAtZero: false,
                grid: { color: '#f0f0f0' },
                border: { display: false }
            }
        },
        elements: {
            point: { radius: 0, hitRadius: 10, hoverRadius: 4 },
            line: { tension: 0.4, borderWidth: 2 }
        }
    };

    const ctxTemp = document.getElementById('tempChart').getContext('2d');
    const tempChart = new Chart(ctxTemp, {
        type: 'line',
        data: { labels: [], datasets: [{ data: [], borderColor: '#ff6384', backgroundColor: 'rgba(255, 99, 132, 0.1)', fill: true }] },
        options: commonOptions
    });

    const ctxHum = document.getElementById('humChart').getContext('2d');
    const humChart = new Chart(ctxHum, {
        type: 'line',
        data: { labels: [], datasets: [{ data: [], borderColor: '#36a2eb', backgroundColor: 'rgba(54, 162, 235, 0.1)', fill: true }] },
        options: commonOptions
    });

    const ctxLight = document.getElementById('lightChart').getContext('2d');
    const lightChart = new Chart(ctxLight, {
        type: 'line',
        data: { labels: [], datasets: [{ data: [], borderColor: '#ffce56', backgroundColor: 'rgba(255, 206, 86, 0.1)', fill: true }] },
        options: commonOptions
    });

    function addData(chart, label, data) {
        chart.data.labels.push(label);
        chart.data.datasets[0].data.push(data);
        if (chart.data.labels.length > maxDataPoints) {
            chart.data.labels.shift();
            chart.data.datasets[0].data.shift();
        }
        chart.update();
    }

    async function fetchMetrics() {
        try {
            const response = await fetch('/api/data');
            const data = await response.json();
            
            // Update Text
            tempVal.textContent = `${data.temperature.toFixed(1)} °C`;
            humVal.textContent = `${data.humidity.toFixed(1)} %`;
            lightVal.textContent = `${data.light}`;

            // Update Charts
            const now = new Date().toLocaleTimeString();
            addData(tempChart, now, data.temperature);
            addData(humChart, now, data.humidity);
            addData(lightChart, now, data.light);
            
        } catch (error) {
            console.error("Error fetching data:", error);
        }
    }

    // Fetch data every 2 seconds
    setInterval(fetchMetrics, 2000);
    fetchMetrics(); 
});

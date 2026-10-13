const metricMap = {
  accuracy: 0.87,
  f1_score: 0.84,
  recall: 0.82,
  precision: 0.86,
};

const reviewChart = new Chart(document.getElementById('reviewChart'), {
  type: 'radar',
  data: {
    labels: ['Accuracy', 'F1', 'Recall', 'Precision'],
    datasets: [
      {
        label: 'Model score',
        data: [metricMap.accuracy, metricMap.f1_score, metricMap.recall, metricMap.precision],
        fill: true,
        backgroundColor: 'rgba(105, 225, 255, 0.22)',
        borderColor: '#69e1ff',
        pointBackgroundColor: '#69e1ff',
        pointBorderColor: '#07111f',
        pointRadius: 4,
      },
    ],
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      r: {
        min: 0,
        max: 1,
        ticks: {
          stepSize: 0.25,
          backdropColor: 'transparent',
          color: '#a7bacd',
        },
        pointLabels: {
          color: '#edf6ff',
        },
        grid: { color: 'rgba(167, 186, 205, 0.18)' },
        angleLines: { color: 'rgba(167, 186, 205, 0.18)' },
      },
    },
    plugins: {
      legend: {
        display: false,
      },
    },
  },
});

const reviewInput = document.getElementById('reviewInput');
const predictBtn = document.getElementById('predictBtn');
const sampleBtn = document.getElementById('sampleBtn');
const predictionBadge = document.getElementById('predictionBadge');
const confidenceFill = document.getElementById('confidenceFill');
const confidenceText = document.getElementById('confidenceText');
const predictionSummary = document.getElementById('predictionSummary');

function analyzeText(text) {
  const lower = text.toLowerCase();
  const posWords = ['excellent', 'great', 'fast', 'reliable', 'easy', 'responsive', 'support', 'love', 'perfect', 'amazing'];
  const negWords = ['slow', 'bad', 'terrible', 'broken', 'late', 'frustrating', 'expensive', 'confusing', 'issue', 'difficult'];

  let score = 0;
  posWords.forEach((word) => {
    if (lower.includes(word)) score += 1;
  });
  negWords.forEach((word) => {
    if (lower.includes(word)) score -= 1;
  });

  if (score > 1) return { label: 'Positive', confidence: 0.88, tone: 'strong positive language tied to quality, speed, and support.' };
  if (score < -1) return { label: 'Negative', confidence: 0.82, tone: 'negative language suggests issues with speed, reliability, or product quality.' };
  return { label: 'Neutral', confidence: 0.67, tone: 'mixed customer sentiment with a balanced but non-committal review.' };
}

async function predictReview() {
  const text = reviewInput.value.trim();
  if (!text) {
    predictionSummary.textContent = 'Please enter a review before predicting sentiment.';
    return;
  }

  let result = analyzeText(text);

  try {
    if (window.ort) {
      const session = await window.ort.InferenceSession.create('./assets/model/aws_reviews_sentiment.onnx');
      const input = new window.ort.Tensor('string', [text], [1]);
      const output = await session.run({ review_text: input });
      const predicted = Array.isArray(output?.[0]) ? output[0][0] : output?.[0];
      if (predicted !== undefined) {
        const val = Number(predicted);
        result = val === 1 ? { label: 'Positive', confidence: 0.9, tone: 'Model predicts a positive customer experience.' } : { label: 'Negative', confidence: 0.85, tone: 'Model predicts a negative review with service or product concerns.' };
      }
    }
  } catch (error) {
    console.warn('Falling back to heuristic sentiment scoring.', error);
  }

  predictionBadge.textContent = result.label;
  predictionBadge.className = `prediction-badge ${result.label.toLowerCase()}`;
  confidenceFill.style.width = `${Math.round(result.confidence * 100)}%`;
  confidenceText.textContent = `${Math.round(result.confidence * 100)}%`;
  predictionSummary.textContent = `This review reads as ${result.label.toLowerCase()} — ${result.tone}`;
}

predictBtn.addEventListener('click', predictReview);
sampleBtn.addEventListener('click', () => {
  reviewInput.value = 'The setup process was frustrating and the dashboard is confusing, but support eventually helped.';
  predictReview();
});

reviewInput.addEventListener('keydown', (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
    predictReview();
  }
});

window.addEventListener('DOMContentLoaded', () => {
  predictionBadge.className = 'prediction-badge positive';
  predictionBadge.textContent = 'Positive';
  confidenceFill.style.width = '88%';
  confidenceText.textContent = '88%';
  predictionSummary.textContent = 'This review contains strong positive language tied to reliability, speed, and support quality.';
});

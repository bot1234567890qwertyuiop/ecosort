# EcoSort

EcoSort is a web application that uses machine learning to classify waste items as either Organic or Recyclable. The application can process images in various formats (including HEIC) and provides confidence scores for its predictions.

## Features

- Image classification (Organic vs Recyclable)
- Support for HEIC image format
- Web interface for easy image upload
- Low confidence predictions are queued for review
- Real-time prediction visualization

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ecosort.git
cd ecosort
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python code/app.py
```

The application will be available at `http://localhost:5000`

## Project Structure

- `code/` - Main application code
  - `app.py` - Flask web application
  - `predict.py` - Prediction logic
  - `templates/` - HTML templates
  - `static/` - Static files (CSS, JS, images)
- `models/` - Trained model files
- `analysis/` - Analysis results and queue
- `predictions/` - Generated prediction visualizations

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
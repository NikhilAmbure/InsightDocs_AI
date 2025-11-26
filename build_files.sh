#!/bin/bash
set -e

echo "Installing dependencies..."
pip install -r requirements.txt --no-cache-dir

echo "Collecting static files..."
python manage.py collectstatic --noinput --clear

# Skip migrations on Vercel - they'll run locally before deploy
echo "Build complete!"
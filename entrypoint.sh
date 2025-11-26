#!/usr/bin/env bash
# exit on error
set -o errexit

# Run migrations (again, in case new ones were added)
python manage.py migrate

# Create superuser if doesn't exist
python manage.py shell << END
from django.contrib.auth import get_user_model

User = get_user_model()

if not User.objects.filter(username='${DJANGO_SUPERUSER_USERNAME}').exists():
    User.objects.create_superuser(
        '${DJANGO_SUPERUSER_USERNAME}',
        '${DJANGO_SUPERUSER_EMAIL}',
        '${DJANGO_SUPERUSER_PASSWORD}'
    )
    print("Superuser created successfully!")
else:
    print("Superuser already exists!")
END

# Start your application
exec daphne InsightDocs_AI.asgi:application --bind 0.0.0.0:$PORT
from django.db import migrations


def seed_subscription_plans(apps, schema_editor):
    SubscriptionPlan = apps.get_model('payments', 'SubscriptionPlan')

    plans_data = [
        {
            "slug": "free",
            "name": "Free",
            "price_inr": 0,
            "price_display": "₹0",
            "duration_days": 30,
            "doc_limit": 10,
            "chat_limit": 500,
            "max_file_size_mb": 10,
            "token_quota": 0,
            "is_active": True,
        },
        {
            "slug": "pro",
            "name": "Pro",
            "price_inr": 199,
            "price_display": "₹199",
            "duration_days": 30,
            "doc_limit": 50,
            "chat_limit": 5000,
            "max_file_size_mb": 50,
            "token_quota": 2000000,
            "is_active": True,
        },
        {
            "slug": "industry",
            "name": "Industry",
            "price_inr": 1999,
            "price_display": "₹1,999",
            "duration_days": 30,
            "doc_limit": 200,
            "chat_limit": 25000,
            "max_file_size_mb": 100,
            "token_quota": 20000000,
            "is_active": True,
        },
    ]

    for data in plans_data:
        plan, created = SubscriptionPlan.objects.get_or_create(
            slug=data["slug"],
            defaults=data
        )
        if not created:
            for field, val in data.items():
                setattr(plan, field, val)
            plan.save()


def remove_subscription_plans(apps, schema_editor):
    pass


class Migration(migrations.Migration):

    dependencies = [
        ('payments', '0005_remove_usercreditaccount_user_and_more'),
    ]

    operations = [
        migrations.RunPython(seed_subscription_plans, remove_subscription_plans),
    ]


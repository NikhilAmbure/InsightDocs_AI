from django.core.management.base import BaseCommand
from payments.models import SubscriptionPlan


class Command(BaseCommand):
    help = "Seed the SubscriptionPlan table with default Free and Pro plans."

    def handle(self, *args, **options):
        plans = [
            {
                "name": "Free",
                "slug": "free",
                "price_inr": 0,
                "price_display": "₹0",
                "duration_days": 30,
                "doc_limit": 10,
                "chat_limit": 500,
                "max_file_size_mb": 10,
                "is_active": True,
            },
            {
                "name": "Pro",
                "slug": "pro",
                "price_inr": 799.00,
                "price_display": "₹799",
                "duration_days": 30,
                "doc_limit": 50,
                "chat_limit": 5000,
                "max_file_size_mb": 50,
                "is_active": True,
            },
            {
                "name": "Enterprise",
                "slug": "enterprise",
                "price_inr": 8299.00,
                "price_display": "₹8,299",
                "duration_days": 30,
                "doc_limit": 9999,
                "chat_limit": 99999,
                "max_file_size_mb": 500,
                "is_active": False,  # Coming soon
            },
        ]

        for plan_data in plans:
            plan, created = SubscriptionPlan.objects.update_or_create(
                slug=plan_data["slug"],
                defaults=plan_data,
            )
            status = "Created" if created else "Updated"
            self.stdout.write(
                self.style.SUCCESS(f"  {status}: {plan.name} (INR {plan.price_inr})")
            )

        self.stdout.write(self.style.SUCCESS("\nSubscription plans seeded successfully!"))

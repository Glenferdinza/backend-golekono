from django.urls import path
from . import views

urlpatterns = [
    path('itinerary/', views.get_itinerary, name='get_itinerary'),
]

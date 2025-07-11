from django.shortcuts import render


def home_page(request):
    return render(request, "interactive_tutor_playground.html")
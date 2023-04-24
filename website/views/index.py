from django.http import HttpResponse
from django.template import loader



__all__ = ['Index']


class Index():
    def main(request):
        template = loader.get_template('index/main.html')
        context = {
            'test': 'hello'
        }
        return HttpResponse(template.render(context, request))
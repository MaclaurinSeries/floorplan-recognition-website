from django.http import HttpResponse
from django.template import loader
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator



__all__ = ['Index']

@method_decorator(csrf_exempt, name='main')
class Index():
    
    def main(request, *args, **kwargs):
        if request.method == 'GET':
            template = loader.get_template('index/main.html')
            context = {
                'test': 'hello'
            }
            return HttpResponse(template.render(context, request))
        elif request.method == 'POST':
            template = loader.get_template('index/main3d.html')
            polygon = request.POST.get("polygon", "")
            graph = request.POST.get("graph", "")

            context = {
                'polygon': polygon,
                'graph': graph
            }
            return HttpResponse(template.render(context, request))
from django.db import models
from django.db.models import Func, F, Count


class data(models.Model):
    obstime = models.DateTimeField(primary_key=True)
    tempr = models.FloatField()
    ap = models.FloatField()
    ws = models.FloatField()
    wd = models.FloatField()
    rh = models.FloatField()
    blizzard = models.FloatField(null=True)

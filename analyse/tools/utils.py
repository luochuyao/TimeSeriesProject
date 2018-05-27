
import os

from django.conf import settings
import filetype

def upload_file_to_media(upload_file):
    upload_file_name = '%s/%s' %(settings.MEDIA_ROOT, upload_file.name)
    with open(upload_file_name, 'wb') as f:
        for frac in upload_file.chunks():
            f.write(frac)


def upload_file(upload_file):

    upload_file_name = '%s/%s' % (settings.MEDIA_ROOT, upload_file.name)

    with open(upload_file_name, 'wb') as f:
        for frac in upload_file.chunks():
            f.write(frac)

    print(upload_file_name)
    print(type(filetype.guess(upload_file_name)))

    os.remove(upload_file_name)
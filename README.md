Starting the server:

Step 1: clone repo onto local machine and enter files

Step 2: Delete the database file using command - rm db.sqlite3

Step 3: Clear migration files using command - find myapp/migrations -type f ! -name '\_\_init\_\_.py' -delete

Step 4: Recreate database using command - python manage.py makemigrations

Step 5: Apply migrations using command - python manage.py migrate

Step 6: Restart the server using command - python manage.py runserver

Common issues:

1) If commands fail to execute use python3 instead of python.

2) In case of hangs or disturbances follow steps 2 to 6 again

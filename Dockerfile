
#specifying the python type
FROM python:3.8-slim

#this is specifyinhg the working directory of the boat
WORKDIR /final project


# Taking everything from the curent directory into the final project inside the docker container

COPY . /finalproject

#running the requirement file

RUN pip install -r requirements.txt

#set a port 
EXPOSE 8501

CMD ["streamlit", "run", "loan_approval_model.py"]
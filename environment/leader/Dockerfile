FROM apache/spark:latest

USER root

# Install OpenSSH server
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/partial/ && \
    apt-get autoclean && \
    apt-get update && \
    apt-get install -y openssh-server python3-pip && \
    mkdir /var/run/sshd && \
    echo 'root:root' | chpasswd && \
    echo "PermitRootLogin yes" >> /etc/ssh/sshd_config && \
    ssh-keygen -A && \
    wget -P /opt/spark/jars \
    -q https://repo1.maven.org/maven2/com/mysql/mysql-connector-j/9.2.0/mysql-connector-j-9.2.0.jar

RUN pip3 install \
    numpy \
    pandas \
    dash

EXPOSE 22 8080 7077 8050

CMD service ssh start && /opt/spark/bin/spark-class org.apache.spark.deploy.master.Master -h spark-leader
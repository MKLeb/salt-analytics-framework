FROM ghcr.io/saltstack/salt-ci-containers/centos-stream:9 as base

ENV LANG=C.UTF-8
ENV LANGUAGE=C.UTF-8
RUN ln -sf /etc/localtime /usr/share/zoneinfo/America/Denver

RUN dnf update -y \
    && dnf upgrade -y \
    && dnf install -y sed vim tmux sudo tree net-tools bind-utils lsof nmap which binutils iputils epel-release procps \
    && dnf install -y --allowerasing curl \
    && dnf install -y multitail supervisor

RUN mkdir -p /etc/supervisor/conf.d/
ADD docker/elastic/conf/supervisord.conf /etc/supervisor/supervisord.conf

RUN rpm --import https://repo.saltproject.io/salt/py3/redhat/9/x86_64/SALT-PROJECT-GPG-PUBKEY-2023.pub \
  && curl -fsSL https://repo.saltproject.io/salt/py3/redhat/9/x86_64/3006.repo | tee /etc/yum.repos.d/salt.repo \
  && dnf install -y salt

COPY ../../dist/salt*.whl /src/
RUN ls -lah /src \
  && /opt/saltstack/salt/salt-pip install /src/salt_analytics_framework*.whl \
  && rm -f /src/*.whl

COPY ../../examples/dist/salt*.whl /src/
RUN ls -lah /src \
  && /opt/saltstack/salt/salt-pip install --find-links /src/ salt-analytics.examples[elasticsearch] \
  && rm -f /src/*.whl


FROM base as master-1

ADD docker/elastic/conf/supervisord.master.conf /etc/supervisor/conf.d/master.conf
ADD docker/elastic/conf/analytics.master.conf /etc/salt/master.d/salt-analytics.conf
RUN mkdir -p /etc/salt/master.d \
  && echo 'id: master-1' > /etc/salt/master.d/id.conf \
  && echo 'open_mode: true' > /etc/salt/master.d/open-mode.conf \
  && dnf install -y salt-master

CMD ["/usr/bin/supervisord","-c","/etc/supervisor/supervisord.conf"]


FROM base as minion-1

ADD docker/elastic/conf/supervisord.minion.conf /etc/supervisor/conf.d/minion.conf
ADD docker/elastic/conf/beacons.conf /etc/salt/minion.d/beacons.conf
ADD docker/elastic/conf/analytics.minion.conf /etc/salt/minion.d/salt-analytics.conf
RUN mkdir -p /etc/salt/minion.d \
  && echo 'id: minion-1' > /etc/salt/minion.d/id.conf \
  && echo 'master: master-1' > /etc/salt/minion.d/master.conf \
  && dnf install -y salt-minion

CMD ["/usr/bin/supervisord","-c","/etc/supervisor/supervisord.conf"]
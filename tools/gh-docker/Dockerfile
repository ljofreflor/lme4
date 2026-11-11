# Isolated git + GitHub CLI environment for the personal account (ljofreflor),
# so it never mixes with the corporate gh/git credentials of the host.
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates curl gnupg \
 && curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
      -o /usr/share/keyrings/githubcli-archive-keyring.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
      > /etc/apt/sources.list.d/github-cli.list \
 && apt-get update && apt-get install -y --no-install-recommends gh \
 && rm -rf /var/lib/apt/lists/*
# credentials live in the volume mounted at /root/.config/gh (gh stores the
# token in its hosts.yml when no keyring is available)
ENV GIT_AUTHOR_NAME="Leonardo Jofré Flor" GIT_AUTHOR_EMAIL="ljofre2146@gmail.com" \
    GIT_COMMITTER_NAME="Leonardo Jofré Flor" GIT_COMMITTER_EMAIL="ljofre2146@gmail.com"
RUN git config --system safe.directory /work \
 && git config --system credential.https://github.com.helper "" \
 && git config --system --add credential.https://github.com.helper "!gh auth git-credential"
WORKDIR /work
ENTRYPOINT ["/bin/bash", "-lc"]

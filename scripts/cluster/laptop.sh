# LAPTOP: shared settings for the scripts that talk to DeltaAI (sourced, not run).
# One ssh "master" connection (one Duo prompt) is reused by every later ssh/rsync for 4 h.
DTAI_HOST="vwudaru@dtai-login.delta.ncsa.illinois.edu"
DTAI_SSH_OPTS=(-o ControlMaster=auto -o "ControlPath=~/.ssh/dtai-%C" -o ControlPersist=4h)
# rsync gets the same options, so whichever command runs first opens the master.
DTAI_RSYNC_SSH="ssh -o ControlMaster=auto -o ControlPath=~/.ssh/dtai-%C -o ControlPersist=4h"

# Image for the "sniffer" sidecar of the Level 8 no-egress check: alpine plus tcpdump (plain alpine has no tcpdump).
# The alpine tag is the release the local alpine image reports in /etc/os-release.
FROM alpine:3.24.2
RUN apk add --no-cache tcpdump

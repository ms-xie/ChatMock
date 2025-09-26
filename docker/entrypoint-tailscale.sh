#!/bin/sh
set -e

# Start the tailscaled daemon in the background.
/usr/local/bin/containerboot &

# Wait a few seconds for the daemon to be ready.
echo "Waiting for tailscaled to start..."
until tailscale status > /dev/null 2>&1; do
  sleep 1
done
echo "Tailscaled is running."

# Check if we are already logged in. `tailscale ip` is a good check,
# as it only succeeds when authenticated.
if tailscale ip -4 > /dev/null 2>&1; then
  echo "Already logged in to Tailscale."
else
  # Not logged in, so start the interactive login process.
  # This will print the URL to the console.
  echo "Starting Tailscale web login..."
  tailscale up
fi

# Once logged in (either now or previously), enable the funnel.
# echo "Enabling Tailscale funnel on port 8000."
# tailscale funnel --bg 8000

# Wait for the background containerboot process to exit.
# This keeps the container running.
wait $!
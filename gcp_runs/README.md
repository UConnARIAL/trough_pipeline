# Globus destination setup on a GCP VM

This guide shows how to set up a **GCP VM as a Globus destination endpoint** using **Globus Connect Personal**, so files can be transferred from an existing HPC Globus endpoint to the VM for further processing.

This workflow is suitable for a **single-user VM** and works well for staging data such as binary masks, model outputs, and intermediate processing files.

---

## Overview

In this setup:

* the **HPC endpoint** is the **source**
* the **GCP VM** is the **destination**
* Globus Connect Personal runs on the VM and exposes a selected destination folder

Recommended destination folder in this example:

```bash
/mnt/data/masks_in
```

---

## 1. Install basic dependencies on the VM

```bash
sudo apt-get update
sudo apt-get install -y wget tar
```

---

## 2. Download and unpack Globus Connect Personal

```bash
cd ~
wget https://downloads.globus.org/globus-connect-personal/linux/stable/globusconnectpersonal-latest.tgz
tar xzf globusconnectpersonal-latest.tgz
cd globusconnectpersonal-*
```

---

## 3. Run first-time setup

```bash
./globusconnectpersonal -setup
```

During setup:

* You will be asked to login to Globus via a given URL, and a code will be generated on the browser that you need to copy and past to proceed with the setup 

* choose a clear endpoint name such as:

```text
presto-pilot-gcp-vm
```

* follow the prompts to complete registration

---

## 4. Start Globus Connect Personal

For an initial test, you can start it in the background with:

```bash
./globusconnectpersonal -start &
```

This is fine for a quick manual test. However, for a more reliable setup that survives logout, terminal closure, and reboot, the recommended approach is the **user systemd service** described later in this guide.

If you want a temporary session-based alternative, you could also run it inside `tmux`, but `systemd --user` is the better long-term option.

---

## 5. Check status

```bash
./globusconnectpersonal -status
```

A healthy setup should show that Globus is connected.

---

## 6. Restrict access to only the destination folder

Create the destination directory:

```bash
mkdir -p /mnt/data/masks_in
```

Edit the allowed path configuration so Globus exposes only that folder:

```bash
printf "%s\n" "/mnt/data/masks_in,0,1" > ~/.globusonline/lta/config-paths
```

This gives read/write access to that directory only.

Restart Globus Connect Personal so the path change takes effect:

```bash
./globusconnectpersonal -stop
./globusconnectpersonal -start &
```

---

## 7. Find the VM endpoint in the Globus web UI

Open the Globus File Manager in your browser.

In the **Collection** search box:

* search for the endpoint name you chose during setup
* for example:

```text
presto-pilot-gcp-vm
```

Then:

* select your **HPC endpoint** as the source
* select the **GCP VM endpoint** as the destination
* browse to:

```bash
/mnt/data/masks_in
```

* start the transfer

If the endpoint does not appear immediately, wait a minute and refresh the web UI.

---

## 8. Recommended transfer target path

Use a dedicated input directory for incoming files, for example:

```bash
/mnt/data/masks_in
```

This keeps transferred data separate from code, configs, and other home-directory files.

---

## 9. Configure reliable background start and auto-start after reboot

The most reliable way to keep Globus Connect Personal running even if your terminal closes is to run it as a **systemd user service**. This is preferable to leaving it attached to a shell, and generally better than using `tmux` for a long-term setup.

Move the extracted directory to a fixed location:

```bash
mv ~/globusconnectpersonal-* ~/.globusconnectpersonal
```

Create a user systemd service:

```bash
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/globusconnectpersonal.service <<'EOF'
[Unit]
Description=Globus Connect Personal

[Service]
ExecStart=%h/.globusconnectpersonal/globusconnectpersonal -start -debug

[Install]
WantedBy=default.target
EOF
```

Start and enable the user service:

```bash
systemctl --user start globusconnectpersonal
systemctl --user enable globusconnectpersonal
systemctl --user status globusconnectpersonal
```

Enable lingering so the user service can continue across logout and start on reboot:

```bash
sudo loginctl enable-linger $USER
```

---

## 10. Useful status and debug commands

Check service status:

```bash
~/.globusconnectpersonal/globusconnectpersonal -status
```

Get trace/debug information:

```bash
~/.globusconnectpersonal/globusconnectpersonal -trace
```

Check systemd user service status:

```bash
systemctl --user status globusconnectpersonal
```

---

## 11. Notes

* This setup is appropriate for a **single-user VM**.
* For a shared or more permanent multi-user system, consider **Globus Connect Server** instead.
* Restricting `config-paths` to a specific folder is recommended rather than exposing the full home directory.
* For large transfers, keep the destination folder dedicated and organized.
* Starting Globus Connect Personal with `&` is acceptable for testing, but it is not the best long-term approach if you want the service to survive logout and reboot.
* `tmux` can be used as a temporary workaround to keep a shell session alive, but a **systemd user service** is the recommended persistent solution.

---

## 12. Example workflow summary

1. Install Globus Connect Personal on the VM
2. Register the VM endpoint with `./globusconnectpersonal -setup`
3. Start the service
4. Restrict the allowed path to `~/masks_in`
5. Search for the VM endpoint name in the Globus web UI
6. Transfer from the HPC endpoint to the VM destination folder
7. Configure the user systemd service for auto-start on reboot

---

## 13. Example destination folder creation

```bash
mkdir -p /mnt/data/masks_in
```

---

## 14. Quick command summary

```bash
sudo apt-get update
sudo apt-get install -y wget tar
cd ~
wget https://downloads.globus.org/globus-connect-personal/linux/stable/globusconnectpersonal-latest.tgz
tar xzf globusconnectpersonal-latest.tgz
cd globusconnectpersonal-*
./globusconnectpersonal -setup
./globusconnectpersonal -start &
mkdir -p /mnt/data/masks_in
printf "%s\n" "/mnt/data/masks_in,0,1" > ~/.globusonline/lta/config-paths
./globusconnectpersonal -stop
./globusconnectpersonal -start &
mv ~/globusconnectpersonal-* ~/.globusconnectpersonal
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/globusconnectpersonal.service <<'EOF'
[Unit]
Description=Globus Connect Personal

[Service]
ExecStart=%h/.globusconnectpersonal/globusconnectpersonal -start -debug

[Install]
WantedBy=default.target
EOF
systemctl --user start globusconnectpersonal
systemctl --user enable globusconnectpersonal
systemctl --user status globusconnectpersonal
sudo loginctl enable-linger $USER
```


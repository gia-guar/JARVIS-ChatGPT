# Copyright (C) 2023 TroubleChute (Wesley Pyburn)
# Licensed under the GNU General Public License v3.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.gnu.org/licenses/gpl-3.0.en.html
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#    
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#    
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# ----------------------------------------
# This script:
# 1. Check if current directory is oobabooga-windows, or oobabooga-windows is in directory
# 2. Run my install script for obabooga/text-generation-webui
# 3. Tells you how to download the vicuna model, and opens the model downloader.
# 4. Run the model downloader (Unless CPU-Only mode was selected in install, in which case the CPU model is downloaded)
# 5. Replace commands in the start-webui.bat file
# 6. Create desktop shortcuts
# 7. Run the webui
# ----------------------------------------

Write-Host "Welcome to TroubleChute's Vicuna installer!" -ForegroundColor Cyan
Write-Host "Vicuna as well as all of its other dependencies and a model should now be installed..." -ForegroundColor Cyan
Write-Host "[Version 2023-04-11]`n`n" -ForegroundColor Cyan

# 1. Check if current directory is oobabooga-windows, or oobabooga-windows is in directory
# If it is, CD back a folder.
$currentDir = (Get-Item -Path ".\" -Verbose).FullName
if ($currentDir -like "*\oobabooga-windows") {
    Set-Location ../
}

$containsFolder = Get-ChildItem -Path ".\" -Directory -Name | Select-String -Pattern "oobabooga-windows"
if ($containsFolder) {
    Write-Host "The 'oobabooga-windows' folder already exists." -ForegroundColor Cyan
    $downloadAgain = Read-Host "Do you want to download it again? (Y/N)"

    if ($downloadAgain -eq "Y" -or $downloadAgain -eq "y") {
        # Perform the download again
        $containsFolder = $False
    }
}

if (-not $containsFolder) {
    Write-Host "I'll start by installing Oobabooga first, then we'll get to the model...`n`n"
    
    #2. Choose CPU or GPU installation
    Write-Host "Do you have an NVIDIA GPU?" -ForegroundColor Cyan
    Write-Host "Enter anything but y or n to skip." -ForegroundColor Yellow

    $choice = Read-Host "Answer (y/n)"

    $skip_model = 1
    $skip_start = 1

    if ($choice -eq "Y" -or $choice -eq "y") {
        Write-Host "Installing GPU & CPU compatible version" -ForegroundColor Cyan
        Write-Host "If this fails, please delete the folder and choose 'N'" -ForegroundColor Cyan
        $gpu = "Yes"
        # 2. Run my install script for obabooga/text-generation-webui
        iex (irm ooba.tc.ht)
    }
    elseif ($choice -eq "N" -or $choice -eq "n") {
        Write-Host "Installing CPU-Only version" -ForegroundColor Cyan
        $gpu = "No"
        # 2. Run my install script for obabooga/text-generation-webui
        iex (irm ooba.tc.ht)
    }

} else {
    # CD into folder anyway
    Set-Location "./oobabooga-windows"
}

function Get-VicunaCPU() {
    # Download CPU model (only the updated one)
    # If downloaded using model downloader, another 8.14 GB download will be run...
    $url = "https://huggingface.co/eachadea/ggml-vicuna-13b-4bit/resolve/main/ggml-vicuna-13b-4bit-rev1.bin"
    $outputPath = "text-generation-webui\models\eachadea_ggml-vicuna-13b-4bit\ggml-vicuna-13b-4bit-rev1.bin"

    # Download the file from the URL
    Write-Host "Downloading: eachadea/ggml-vicuna-13b-4bit (CPU model)" -ForegroundColor Cyan
    Get-Aria2File -Url $url -OutputPath $outputPath
    Write-Host "`nDone!`n"
}
function Get-VicunaGPU() {
    # Download GPU/CUDA model
    $blob = "https://huggingface.co/anon8231489123/vicuna-13b-GPTQ-4bit-128g/resolve/main"
    $outputPath = "text-generation-webui\models\anon8231489123_vicuna-13b-GPTQ-4bit-128g"

    # Download the file from the URL
    Write-Host "Downloading: anon8231489123/vicuna-13b-GPTQ-4bit-128g (GPU/CUDA model)" -ForegroundColor Cyan
    $files = @(
        "vicuna-13b-4bit-128g.safetensors",
        "tokenizer_config.json",
        "tokenizer.model",
        "special_tokens_map.json",
        "pytorch_model.bin.index.json",
        "generation_config.json",
        "config.json"
    )

    Get-Aria2Files -Url $blob -OutputPath $outputPath -Files $files
    Write-Host "`nDone!`n"
}

# Allow importing remote functions
iex (irm Import-RemoteFunction.tc.ht)
Import-FunctionIfNotExists -Command Get-Aria2File -ScriptUri "File-DownloadMethods.tc.ht"

# Create the output folder if it does not exist
New-Item -ItemType Directory -Force -Path (Split-Path -Parent "text-generation-webui\models\eachadea_ggml-vicuna-13b-4bit") | Out-Null
if (-not $?) {
    Write-Error "Failed to create directory."
}

if ($gpu -eq "No") {
    Get-VicunaCPU
} else {
    Write-Host "`n`nPick which models to download:" -ForegroundColor Cyan
    Write-Host -NoNewline "CPU (7.5GB): " -ForegroundColor Red
    Write-Host "1" -ForegroundColor Green
    Write-Host -NoNewline "GPU [Nvidia] (6.9GB): " -ForegroundColor Red
    Write-Host "2" -ForegroundColor Green
    Write-Host -NoNewline "CPU + GPU [Nvidia] (14.4GB): " -ForegroundColor Red
    Write-Host "3" -ForegroundColor Green
    
    $num = Read-Host "Enter a number"
    if ($num -eq "1") {
        Get-VicunaCPU
    } elseif ($num -eq "2") {
        Get-VicunaGPU
    } elseif ($num -eq "3") {
        Get-VicunaCPU
        Get-VicunaGPU
    }
}

# 5. Replace commands in the start-webui.bat file
# Create CPU and GPU versions
Copy-Item "start-webui.bat" "start-webui-vicuna.bat"

if (-not ($gpu -eq "No")) {
    (Get-Content -Path "start-webui-vicuna.bat") | ForEach-Object {
        $_ -replace
            'call python server\.py --auto-devices --cai-chat',
            'call python server.py --auto-devices --chat --model anon8231489123_vicuna-13b-GPTQ-4bit-128g --wbits 4 --groupsize 128'
    } | Set-Content -Path "start-webui-vicuna-gpu.bat"
}

(Get-Content -Path "start-webui-vicuna.bat") | ForEach-Object {
    $_ -replace
        'call python server\.py --auto-devices --cai-chat',
        'call python server.py --auto-devices --chat --model eachadea_ggml-vicuna-13b-4bit'
} | Set-Content -Path "start-webui-vicuna.bat"

# 6. Create desktop shortcuts
if ($gpu -eq "No") {
    Write-Host "`n`nCreate desktop shortcuts for 'Vicuna (CPU)'" -ForegroundColor Cyan
} else {
    Write-Host "`n`nCreate desktop shortcuts for 'Vicuna' and 'Vicuna (CPU)'" -ForegroundColor Cyan
}
$shortcuts = Read-Host "Do you want desktop shortcuts? (Y/N)"

if ($shortcuts -eq "Y" -or $shortcuts -eq "y") {
    iex (irm Import-RemoteFunction.tc.ht) # Get RemoteFunction importer
    Import-RemoteFunction -ScriptUri "https://New-Shortcut.tc.ht" # Import function to create a shortcut
    
    Write-Host "Downloading Vicuna icon..."
    Invoke-WebRequest -Uri 'https://tc.ht/PowerShell/AI/vicuna.ico' -OutFile 'vicuna.ico'
    if (-not ($gpu -eq "No")) {
        Write-Host "`nCreating shortcuts on desktop..." -ForegroundColor Cyan
        $shortcutName = "Vicuna oobabooga"
        $targetPath = "start-webui-vicuna-gpu.bat"
        $IconLocation = 'vicuna.ico'
        New-Shortcut -ShortcutName $shortcutName -TargetPath $targetPath -IconLocation $IconLocation
    }

    $shortcutName = "Vicuna (CPU) oobabooga"
    $targetPath = "start-webui-vicuna.bat"
    $IconLocation = 'vicuna.ico'
    New-Shortcut -ShortcutName $shortcutName -TargetPath $targetPath -IconLocation $IconLocation
    
}

# 7. Run the webui
if ($gpu -eq "No") {
    Start-Process ".\start-webui-vicuna.bat"
} else {
    # Ask user if they want to launch the CPU or GPU version
    Write-Host "`n`nEnter 1 to launch CPU version, or 2 to launch GPU version" -ForegroundColor Cyan
    
    $choice = Read-Host "1 (CPU) or 2 (GPU)"
    
    if ($choice -eq "1") {
        Start-Process ".\start-webui-vicuna.bat"
    }
    elseif ($choice -eq "2") {
        Start-Process ".\start-webui-vicuna-gpu.bat"
    }
    else {
        Write-Host "Invalid choice. Please enter 1 or 2."
    }
}

@echo off
powershell.exe -ExecutionPolicy Bypass -File "%~dp0build.ps1"
if %errorlevel% neq 0 pause

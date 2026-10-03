@echo off
title Vercel Production Deployment
echo ========================================================
echo         Vercel Production Deployment
echo ========================================================
echo.
cd /d "c:\Users\User\OneDrive\Desktop\02 Projects\object detection app\client"

echo [Step 1] Logging in to Vercel...
echo Please choose your login method in the prompt below:
echo.
call vercel login

echo.
echo [Step 2] Deploying React Client to Vercel...
call vercel --prod

echo.
echo ========================================================
echo Done! Your live Vercel URL is displayed above.
echo ========================================================
pause

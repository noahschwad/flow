import './globals.css'

export const metadata = {
  title: 'murm',
  description: 'Fluid flow simulation',
}

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <head>
        <meta name="viewport" content="width=device-width, height=device-height, interactive-widget=resizes-content, shrink-to-fit=0, user-scalable=no, minimum-scale=1.0, maximum-scale=1.0" />
        <script defer src="https://s.holtsetio.com/script.js" data-website-id="cb36fa92-2381-4031-8f81-f430a473156d"></script>
      </head>
      <body>{children}</body>
    </html>
  )
}


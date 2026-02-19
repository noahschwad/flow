'use client'

// Force dynamic rendering for WebGPU
export const dynamic = 'force-dynamic'

import { useEffect, useRef, useState, createContext, useContext } from 'react'
import * as THREE from "three/webgpu"
import App from "../src/app"

THREE.ColorManagement.enabled = true

// Create context for app instance
const AppContext = createContext(null)

export const useApp = () => useContext(AppContext)

export default function Home() {
  const containerRef = useRef(null)
  const appRef = useRef(null)
  const rendererRef = useRef(null)
  const clockRef = useRef(null)
  const animationFrameRef = useRef(null)
  const initializedRef = useRef(false) // Prevent double initialization in StrictMode
  
  const [loadingProgress, setLoadingProgress] = useState(0)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState(null)

  const updateLoadingProgressBar = async (frac, delay = 0) => {
    return new Promise(resolve => {
      setLoadingProgress(frac)
      if (delay === 0) {
        resolve()
      } else {
        setTimeout(resolve, delay)
      }
    })
  }

  const createRenderer = () => {
    const renderer = new THREE.WebGPURenderer({
      //forceWebGL: true,
      //antialias: true,
    })
    renderer.setPixelRatio(window.devicePixelRatio)
    renderer.setSize(window.innerWidth, window.innerHeight)
    renderer.outputColorSpace = THREE.SRGBColorSpace
    return renderer
  }

  const handleError = (msg) => {
    setIsLoading(false)
    setError(msg)
  }

  useEffect(() => {
    // Prevent double initialization in React StrictMode
    if (initializedRef.current) {
      return
    }
    initializedRef.current = true

    // Check WebGPU support
    if (!navigator.gpu) {
      handleError("Your device does not support WebGPU.")
      return
    }

    const init = async () => {
      try {
        const renderer = createRenderer()
        await renderer.init()

        if (!renderer.backend.isWebGPUBackend) {
          handleError("Couldn't initialize WebGPU. Make sure WebGPU is supported by your Browser!")
          return
        }

        rendererRef.current = renderer

        // Append renderer to container
        if (containerRef.current) {
          containerRef.current.appendChild(renderer.domElement)
        }

        const app = new App(renderer)
        appRef.current = app
        await app.init(updateLoadingProgressBar)

        // Hide loading UI
        setIsLoading(false)

        // Setup resize handler
        const handleResize = () => {
          if (renderer && app) {
            renderer.setSize(window.innerWidth, window.innerHeight)
            app.resize(window.innerWidth, window.innerHeight)
          }
        }
        window.addEventListener("resize", handleResize)
        handleResize()

        // Setup animation loop
        const clock = new THREE.Clock()
        clockRef.current = clock

        const animate = async () => {
          if (app && renderer) {
            const delta = clock.getDelta()
            const elapsed = clock.getElapsedTime()
            await app.update(delta, elapsed)
            animationFrameRef.current = requestAnimationFrame(animate)
          }
        }
        animationFrameRef.current = requestAnimationFrame(animate)

        // Cleanup function
        return () => {
          initializedRef.current = false
          window.removeEventListener("resize", handleResize)
          if (animationFrameRef.current) {
            cancelAnimationFrame(animationFrameRef.current)
          }
          if (renderer && containerRef.current && renderer.domElement && containerRef.current.contains(renderer.domElement)) {
            containerRef.current.removeChild(renderer.domElement)
          }
          if (renderer) {
            renderer.dispose()
          }
          if (app) {
            // Clean up app resources if needed
            appRef.current = null
          }
        }
      } catch (err) {
        console.error(err)
        handleError(err.message || "An error occurred during initialization")
      }
    }

    init()
  }, [])

  return (
    <AppContext.Provider value={appRef.current || null}>
      <div id="container" ref={containerRef} style={{ width: '100%', height: '100%', display: 'block' }}>
        {isLoading && (
          <div id="veil" style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            backgroundColor: 'black',
            opacity: isLoading ? 1 : 0,
            pointerEvents: 'none',
            transition: 'opacity 1s ease-in-out'
          }}>
            <div id="progress-bar" style={{
              position: 'absolute',
              width: '200px',
              height: '5px',
              left: '50vw',
              top: '50vh',
              transition: 'opacity 0.2s ease',
              transform: 'translateX(-50%) translateY(-50%)',
              backgroundColor: '#333'
            }}>
              <div id="progress" style={{
                position: 'absolute',
                width: `${loadingProgress * 200}px`,
                height: '5px',
                left: 0,
                top: 0,
                transition: 'width 0.2s ease',
                backgroundColor: '#848484'
              }}></div>
            </div>
            {error && (
              <div id="error" style={{
                position: 'absolute',
                left: '50vw',
                top: '50vh',
                transform: 'translateX(-50%) translateY(-50%)',
                color: '#FFFFFF',
                pointerEvents: 'auto'
              }}>
                Error: {error}
              </div>
            )}
          </div>
        )}
        {error && !isLoading && (
          <div id="error" style={{
            position: 'absolute',
            left: '50vw',
            top: '50vh',
            transform: 'translateX(-50%) translateY(-50%)',
            color: '#FFFFFF',
            pointerEvents: 'auto'
          }}>
            Error: {error}
          </div>
        )}
      </div>
    </AppContext.Provider>
  )
}


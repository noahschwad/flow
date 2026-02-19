import { defineConfig } from 'vite'

export default defineConfig({
    base: './',
    assetsInclude: ['**/*.hdr'],
    server: {
        port: 1234,
    },
});
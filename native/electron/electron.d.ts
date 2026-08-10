declare module "electron" {
  export const app: {
    whenReady(): Promise<void>;
    quit(): void;
  };
}

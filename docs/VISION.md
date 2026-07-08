# Vision: FrameLanguageLM

- **user**: Pepe
- **date**: 2026-07-08
- **workflow**: construction

## Core need

Poder encontrar películas y series que de verdad se alineen con lo que me gusta — descubrir las que querría ver pero no conozco, resolviendo el "martes por la noche: ¿y yo qué veo ahora?".

## Success criteria

- [ ] Con mi perfil real (FilmAffinity/Netflix): del top-10 de huecos que proponga, me gustan más de 5 ("me las vería")
- [ ] El modelo produce al menos 50 recomendaciones de calidad, para que el usuario tenga donde elegir
- [ ] Interfaz web pública: conectar perfil de FilmAffinity/Letterboxd o similar (auth o solo URL del perfil) y recibir la lista
- [ ] Búsqueda fina funcionando: no solo "la siguiente película", también "la siguiente comedia", "la siguiente de estos directores", "la siguiente de este país"

## Red lines

- Prácticamente gratuito de operar — no gastar dinero si se puede evitar
- El usuario medio NO debe tener que instalar nada
- No debe convertirse en un problema/carga continua para Pepe

## Deliverables

- Modelo publicado en HuggingFace
- Repo público en GitHub: proyecto completo, metodología y sistema explicados, con vía para que cualquiera descargue sus datos y se los dé al modelo
- Página web pública para usarlo
- Post en LinkedIn + presentación pública en Reddit y similares
- Documento de caso de estudio listo para la futura página personal (portfolio)

## Context

- Plazos: cuanto antes mejor
- Dependencias: si se puede, ninguna
- FrameLanguageLM es parte de un proyecto mayor: LittleMeLLM (es uno de sus modelos)
- Naturaleza del proyecto: pieza bonita de portfolio que además se pueda vender si hay suerte y hay usuarios
